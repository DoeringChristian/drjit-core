/*
    src/vulkan_ts.cpp -- ThreadState specialization for the Vulkan backend

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(DRJIT_ENABLE_VULKAN)

#include "vulkan_ts.h"
#include "vulkan.h"
#include "vulkan_api.h"
#include "log.h"
#include "malloc.h"
#include "io.h"
#include "history.h"

#include <cstring>

/// A submitted command buffer, released once the device has executed it
struct VulkanCommandBuffer {
    VkDevice device;
    VkCommandPool pool;
    VkCommandBuffer cb;
};

static void jitc_vulkan_release_cmdbuf(void *payload) {
    VulkanCommandBuffer *c = (VulkanCommandBuffer *) payload;

    /* The command pool is shared by all threads using the device */ {
        lock_guard guard(state.lock);
        vkFreeCommandBuffers(c->device, c->pool, 1, &c->cb);
    }

    delete c;
}

/// Make the writes of all preceding commands (including those of earlier
/// submissions to the queue) visible to the given stage and access types
static void jitc_vulkan_barrier(VkCommandBuffer cb, VkPipelineStageFlags stage,
                                VkAccessFlags access) {
    VkMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = access;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, stage, 0, 1,
                         &barrier, 0, nullptr, 0, nullptr);
}

VulkanThreadState::~VulkanThreadState() {
    // Wait for any queued computation to finish
    flush(/* wait = */ true);
}

// ============================================================================
//  Command buffer lifecycle
// ============================================================================

void *VulkanThreadState::ensure_cmdbuf() {
    if (!vulkan_cb) {
        VkCommandBufferAllocateInfo alloc_info {};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = (VkCommandPool) vulkan_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        vulkan_check(vkAllocateCommandBuffers(
            (VkDevice) vulkan_device, &alloc_info, (VkCommandBuffer *) &vulkan_cb));

        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vulkan_check(vkBeginCommandBuffer((VkCommandBuffer) vulkan_cb,
                                          &begin_info));

        // All kernels share the pipeline layout, hence the texture heap stays
        // bound across pipelines
        const VulkanDevice &dev = state.vulkan_devices[device];
        VkDescriptorSet heap = (VkDescriptorSet) dev.heap;
        vkCmdBindDescriptorSets((VkCommandBuffer) vulkan_cb,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                (VkPipelineLayout) dev.layout, 0, 1, &heap, 0,
                                nullptr);
    }

    jitc_vulkan_barrier((VkCommandBuffer) vulkan_cb,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    return vulkan_cb;
}

void VulkanThreadState::flush(bool wait) {
    if (vulkan_cb) {
        // Submit the current command buffer
        VkCommandBuffer cb = (VkCommandBuffer) vulkan_cb;
        VkSemaphore timeline = (VkSemaphore) vulkan_timeline;

        // The host reads shared memory once the submission has finished
        jitc_vulkan_barrier(cb, VK_PIPELINE_STAGE_HOST_BIT,
                            VK_ACCESS_HOST_READ_BIT);
        vulkan_check(vkEndCommandBuffer(cb));

        uint64_t value = ++state.vulkan_devices[device].submitted;

        VkTimelineSemaphoreSubmitInfo timeline_info {};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &value;

        VkSubmitInfo submit_info {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.pNext = &timeline_info;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cb;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &timeline;
        vulkan_check(vkQueueSubmit((VkQueue) vulkan_queue, 1, &submit_info,
                                   VK_NULL_HANDLE));

        // Release the command buffer once the device has executed it
        jitc_vulkan_enqueue_callback(
            device, jitc_vulkan_release_cmdbuf,
            new VulkanCommandBuffer{ (VkDevice) vulkan_device,
                                     (VkCommandPool) vulkan_pool, cb });
        vulkan_cb = nullptr;
        vulkan_last = value;
    }

    // Handle deferred frees once the work submitted so far has finished
    if (void *batch = take_deferred_free())
        jitc_vulkan_enqueue_callback(device, jitc_malloc_release_batch, batch);

    // Return released buffers to the memory cache
    if (!free_next.empty()) {
        for (const auto &e : free_next)
            jitc_malloc_release(e.first, e.second);
        free_next.clear();
    }

    if (wait)
        vulkan_wait(vulkan_device, vulkan_timeline, vulkan_last);
}

void VulkanThreadState::flush_deferred_free() {
    if (!free_later.empty() || !free_next.empty())
        flush(/* wait = */ false);
}

void VulkanThreadState::barrier() {
    // barrier() is called after enqueueing kernels in jitc_eval(). The Vulkan
    // backend interprets it as a hint to submit the command buffer to the GPU
    // to overlap GPU execution with subsequent tracing steps.
    flush(/* wait = */ false);
}

void VulkanThreadState::enqueue_host_func(void (*callback)(void *),
                                          void *payload) {
    flush(/* wait = */ false);
    jitc_vulkan_enqueue_callback(device, callback, payload);
}

// ============================================================================
//  Kernel launch
// ============================================================================

Task *VulkanThreadState::launch(Kernel kernel, KernelKey & /*key*/,
                                XXH128_hash_t /*hash*/, uint32_t size,
                                std::vector<void *> &kernel_params,
                                const std::vector<uint32_t> & /*kernel_param_ids*/,
                                KernelHistoryEntry *kernel_history_entry) {
    const VulkanDevice &dev = state.vulkan_devices[device];

    // Larger kernels are processed by a grid-stride loop (see
    // jitc_vulkan_assemble())
    uint32_t groups = std::min(
        (size + dev.block_size - 1) / dev.block_size,
        dev.max_group_count);

    // Upload the parameter array to device memory. The kernel receives its
    // device address through a push constant.
    size_t params_bytes = kernel_params.size() * sizeof(void *);
    void *params = jitc_malloc(JitBackend::Vulkan, params_bytes);
    if (params_bytes <= 65536) {
        size_t offset = 0;
        VulkanBuffer *buf =
            (VulkanBuffer *) jitc_vulkan_find_buffer(params, &offset);
        vkCmdUpdateBuffer((VkCommandBuffer) ensure_cmdbuf(),
                          (VkBuffer) buf->buffer, offset, params_bytes,
                          kernel_params.data());
    } else {
        // vkCmdUpdateBuffer() is limited to 64 KiB, copy larger arrays via a
        // staging buffer
        void *staging =
            jitc_malloc(JitBackend::Vulkan, params_bytes, /*shared=*/true);
        std::memcpy(staging, kernel_params.data(), params_bytes);
        memcpy_async(params, staging, params_bytes);
        jitc_free(staging);
    }

    VkCommandBuffer cb = (VkCommandBuffer) ensure_cmdbuf();
    uint64_t params_addr = (uint64_t) (uintptr_t) params;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      (VkPipeline) kernel.vulkan.pipeline);
    vkCmdPushConstants(cb, (VkPipelineLayout) dev.layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint64_t),
                       &params_addr);
    vkCmdDispatch(cb, groups, 1, 1);

    // Later reuses of the memory are ordered after this launch (see
    // ThreadState::free_next)
    jitc_free(params);

    if (kernel_history_entry) {
        kernel_history_entry->size = size;
        jitc_kernel_history_append(*kernel_history_entry);
    }

    return nullptr;
}

// ============================================================================
//  Memory operations
// ============================================================================

void VulkanThreadState::memset_async(void *ptr, uint32_t size, uint32_t isize,
                                     const void *src) {
    if (size == 0)
        return;

    jitc_trace("jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
               (uintptr_t) ptr, isize, size);

    size_t offset = 0;
    VulkanBuffer *buf = (VulkanBuffer *) jitc_vulkan_find_buffer(ptr, &offset);
    size_t bytes = (size_t) size * isize;

    // vkCmdFillBuffer() replicates a 32-bit pattern
    uint32_t value;
    switch (isize) {
        case 1: value = *(const uint8_t *) src * 0x01010101u; break;
        case 2: value = *(const uint16_t *) src * 0x00010001u; break;
        case 4: value = *(const uint32_t *) src; break;
        default: value = 0; break;
    }

    // ... into 4-byte aligned ranges. GPU allocations are padded to multiples
    // of 64 bytes, hence a fill starting at an allocation may round up.
    if (offset == 0)
        bytes = (bytes + 3) / 4 * 4;

    if (isize > 4 || offset % 4 != 0 || bytes % 4 != 0)
        jitc_raise("jit_memset_async(): %u-byte patterns at offset %zu are not "
                   "yet supported by the Vulkan backend.", isize, offset);

    vkCmdFillBuffer((VkCommandBuffer) ensure_cmdbuf(), (VkBuffer) buf->buffer,
                    offset, bytes, value);
}

void VulkanThreadState::memcpy(void *dst, const void *src, size_t size) {
    size_t off = 0;
    bool src_dev = jitc_vulkan_find_buffer((void *) src, &off) != nullptr;
    bool dst_dev = jitc_vulkan_find_buffer(dst, &off) != nullptr;

    if (src_dev && dst_dev) {
        // Device -> device: enqueue a copy and wait for it to finish
        memcpy_async(dst, src, size);
        flush(/* wait = */ true);
    } else if (!src_dev && !dst_dev) {
        // Host -> host: a plain synchronous copy
        std::memcpy(dst, src, size);
    } else {
        // Exactly one endpoint is host memory: stage through a shared buffer
        // and wait. (jitc_var_mem_copy() handles the asynchronous upload path.)
        void *staging = jitc_malloc(JitBackend::Vulkan, size, /*shared=*/true);
        if (!src_dev) {
            // Host -> device upload
            std::memcpy(staging, src, size);
            memcpy_async(dst, staging, size);
            flush(/* wait = */ true);
        } else {
            // Device -> host readback
            memcpy_async(staging, src, size);
            flush(/* wait = */ true);
            std::memcpy(dst, staging, size);
        }
        jitc_free(staging);
    }
}

void VulkanThreadState::memcpy_async(void *dst, const void *src, size_t size) {
    if (size == 0)
        return;

    size_t src_off = 0, dst_off = 0;
    VulkanBuffer
        *src_buf = (VulkanBuffer *) jitc_vulkan_find_buffer((void *) src, &src_off),
        *dst_buf = (VulkanBuffer *) jitc_vulkan_find_buffer(dst, &dst_off);

    if (unlikely(!src_buf || !dst_buf))
        jitc_fail("VulkanThreadState::memcpy_async(): both endpoints must be "
                  "Vulkan buffers (src=" DRJIT_PTR ", dst=" DRJIT_PTR
                  ", size=%zu); the caller must stage host memory through a "
                  "shared buffer.", (uintptr_t) src, (uintptr_t) dst, size);

    VkBufferCopy region { src_off, dst_off, size };
    vkCmdCopyBuffer((VkCommandBuffer) ensure_cmdbuf(),
                    (VkBuffer) src_buf->buffer, (VkBuffer) dst_buf->buffer, 1,
                    &region);
}

void VulkanThreadState::poke(void *dst, const void *src, uint32_t size) {
    jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);
    // A single-element store is a one-element memset
    memset_async(dst, /*size=*/1, /*isize=*/size, src);
}

// ============================================================================
//  Utility kernels (see resources/vulkan_kernels.comp)
// ============================================================================

void jitc_vulkan_launch_utility(VulkanThreadState *ts, VulkanKernel kernel,
                                const void *params, uint32_t params_size,
                                uint32_t size) {
    const VulkanDevice &dev = state.vulkan_devices[ts->device];
    uint32_t groups = std::min((size + dev.block_size - 1) / dev.block_size,
                               dev.max_group_count);

    VkCommandBuffer cb = (VkCommandBuffer) ts->ensure_cmdbuf();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      (VkPipeline) dev.pipelines[(uint32_t) kernel]);
    vkCmdPushConstants(cb, (VkPipelineLayout) dev.layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, params_size, params);
    vkCmdDispatch(cb, groups, 1, 1);
}

void VulkanThreadState::aggregate(void *dst, AggregationEntry *agg,
                                  uint32_t size) {
    if (size == 0)
        return;

    jitc_log(InfoSym, "jit_aggregate(" DRJIT_PTR " -> " DRJIT_PTR
             ", size=%u)", (uintptr_t) agg, (uintptr_t) dst, size);

    // The caller allocates ``agg`` in shared memory, which the kernel reads
    // in place
    struct { uint64_t dst, agg; uint32_t size; } params {
        (uint64_t) (uintptr_t) dst, jitc_vulkan_device_address(agg), size
    };

    jitc_vulkan_launch_utility(this, VulkanKernel::Aggregate, &params,
                               sizeof(params), size);
}

// ============================================================================
//  Operations that are not yet supported
// ============================================================================

void VulkanThreadState::block_reduce(VarType, ReduceOp, uint32_t, uint32_t,
                                     const void *, void *) {
    jitc_raise("jit_block_reduce(): not yet supported by the Vulkan backend.");
}

void VulkanThreadState::block_prefix_reduce(VarType, ReduceOp, uint32_t,
                                            uint32_t, bool, bool, const void *,
                                            void *) {
    jitc_raise("jit_block_prefix_reduce(): not yet supported by the Vulkan "
               "backend.");
}

void VulkanThreadState::reduce_dot(VarType, const void *, const void *,
                                   uint32_t, void *) {
    jitc_raise("jit_reduce_dot(): not yet supported by the Vulkan backend.");
}

void VulkanThreadState::batched_gemm(VarType, bool, bool, uint32_t, uint32_t,
                                     uint32_t, const GemmBatch *, const void *,
                                     const void *, void *) {
    jitc_raise("jit_batched_gemm(): not yet supported by the Vulkan backend.");
}

uint32_t VulkanThreadState::compress(const uint8_t *, uint32_t, uint32_t *) {
    jitc_raise("jit_compress(): not yet supported by the Vulkan backend.");
}

uint32_t VulkanThreadState::block_mkperm(const uint32_t *, uint32_t, uint32_t,
                                         uint32_t, uint32_t *, uint32_t *) {
    jitc_raise("jit_block_mkperm(): not yet supported by the Vulkan backend.");
}

void VulkanThreadState::coop_vec_pack(uint32_t, const void *,
                                      const MatrixDescr *, void *,
                                      const MatrixDescr *) {
    jitc_raise("jit_coop_vec_pack(): not yet supported by the Vulkan backend.");
}

#endif // defined(DRJIT_ENABLE_VULKAN)
