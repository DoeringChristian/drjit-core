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

/// Raise an error for ThreadState operations that the Vulkan backend lacks
[[noreturn]] static void jitc_vulkan_unsupported(const char *name) {
    jitc_raise("VulkanThreadState::%s(): not yet supported by the Vulkan "
               "backend.", name);
}

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
    }
    return vulkan_cb;
}

void VulkanThreadState::flush(bool wait) {
    if (vulkan_cb) {
        // Submit the current command buffer
        VkCommandBuffer cb = (VkCommandBuffer) vulkan_cb;
        VkSemaphore timeline = (VkSemaphore) vulkan_timeline;
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
//  Operations that are not yet supported
// ============================================================================

Task *VulkanThreadState::launch(Kernel, KernelKey &, XXH128_hash_t, uint32_t,
                                std::vector<void *> &,
                                const std::vector<uint32_t> &,
                                KernelHistoryEntry *) {
    jitc_vulkan_unsupported("launch");
}

void VulkanThreadState::memset_async(void *, uint32_t, uint32_t, const void *) {
    jitc_vulkan_unsupported("memset_async");
}

void VulkanThreadState::block_reduce(VarType, ReduceOp, uint32_t, uint32_t,
                                     const void *, void *) {
    jitc_vulkan_unsupported("block_reduce");
}

void VulkanThreadState::block_prefix_reduce(VarType, ReduceOp, uint32_t,
                                            uint32_t, bool, bool, const void *,
                                            void *) {
    jitc_vulkan_unsupported("block_prefix_reduce");
}

void VulkanThreadState::reduce_dot(VarType, const void *, const void *,
                                   uint32_t, void *) {
    jitc_vulkan_unsupported("reduce_dot");
}

void VulkanThreadState::batched_gemm(VarType, bool, bool, uint32_t, uint32_t,
                                     uint32_t, const GemmBatch *, const void *,
                                     const void *, void *) {
    jitc_vulkan_unsupported("batched_gemm");
}

uint32_t VulkanThreadState::compress(const uint8_t *, uint32_t, uint32_t *) {
    jitc_vulkan_unsupported("compress");
}

uint32_t VulkanThreadState::block_mkperm(const uint32_t *, uint32_t, uint32_t,
                                         uint32_t, uint32_t *, uint32_t *) {
    jitc_vulkan_unsupported("block_mkperm");
}

void VulkanThreadState::memcpy(void *, const void *, size_t) {
    jitc_vulkan_unsupported("memcpy");
}

void VulkanThreadState::memcpy_async(void *, const void *, size_t) {
    jitc_vulkan_unsupported("memcpy_async");
}

void VulkanThreadState::poke(void *, const void *, uint32_t) {
    jitc_vulkan_unsupported("poke");
}

void VulkanThreadState::aggregate(void *, AggregationEntry *, uint32_t) {
    jitc_vulkan_unsupported("aggregate");
}

void VulkanThreadState::coop_vec_pack(uint32_t, const void *,
                                      const MatrixDescr *, void *,
                                      const MatrixDescr *) {
    jitc_vulkan_unsupported("coop_vec_pack");
}

#endif // defined(DRJIT_ENABLE_VULKAN)
