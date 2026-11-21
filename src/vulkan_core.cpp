/*
    src/vulkan_core.cpp -- Vulkan device init and shutdown

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(DRJIT_ENABLE_VULKAN)

#include "vulkan.h"
#include "vulkan_api.h"
#include "vulkan_asm.h"
#include "vulkan_ts.h"
#include "vulkan_tex.h"
#include "buffer_map.h"
#include "internal.h"
#include "eval.h"
#include "unit.h"
#include "io.h"
#include "log.h"
#include "resources/vulkan_kernels.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <thread>
#include <string_view>

static VkInstance jitc_vulkan_instance = nullptr;

/// SPIRV-Tools context used to assemble kernels
static spv_context jitc_vulkan_spirv_context = nullptr;

/// Memory properties of each entry of ``state.vulkan_devices``
static std::vector<VkPhysicalDeviceMemoryProperties> vulkan_memory_properties;

// ============================================================================
// Buffer API
// ============================================================================

/// Return the index of a memory type in ``type_bits`` with the ``required``
/// properties, preferring one that also has the ``preferred`` properties.
/// Returns -1 if there is no such memory type.
static int vulkan_memory_type(const VkPhysicalDeviceMemoryProperties &props,
                              uint32_t type_bits,
                              VkMemoryPropertyFlags required,
                              VkMemoryPropertyFlags preferred) {
    int result = -1;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags flags = props.memoryTypes[i].propertyFlags;
        if (!(type_bits & (1u << i)) || (flags & required) != required)
            continue;
        if ((flags & preferred) == preferred)
            return (int) i;
        if (result < 0)
            result = (int) i;
    }
    return result;
}

void *vulkan_buffer_new(int device, size_t size, bool shared, void **ptr_out) {
    VkDevice dev = (VkDevice) state.vulkan_devices[device].device;

    VkBufferCreateInfo buffer_info {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(dev, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
        return nullptr;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, buffer, &req);

    // Shared memory is read back by the host, hence prefer cached memory
    int type = shared
        ? vulkan_memory_type(vulkan_memory_properties[device],
                             req.memoryTypeBits,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        : vulkan_memory_type(vulkan_memory_properties[device],
                             req.memoryTypeBits,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);

    VkMemoryAllocateFlagsInfo flags_info {};
    flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo alloc_info {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = &flags_info;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = (uint32_t) type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (type < 0 ||
        vkAllocateMemory(dev, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(dev, buffer, nullptr);
        return nullptr;
    }

    vulkan_check(vkBindBufferMemory(dev, buffer, memory, 0));

    if (shared) {
        vulkan_check(vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, ptr_out));
    } else {
        VkBufferDeviceAddressInfo address_info {};
        address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        address_info.buffer = buffer;
        *ptr_out = (void *) (uintptr_t) vkGetBufferDeviceAddress(dev, &address_info);
    }

    return new VulkanBuffer { dev, buffer, memory };
}

void *jitc_vulkan_image_memory(int device, void *image, size_t *size_out) {
    VkDevice dev = (VkDevice) state.vulkan_devices[device].device;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, (VkImage) image, &req);
    int type = vulkan_memory_type(vulkan_memory_properties[device],
                                  req.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);

    VkMemoryAllocateInfo alloc_info {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = (uint32_t) type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (type < 0 ||
        vkAllocateMemory(dev, &alloc_info, nullptr, &memory) != VK_SUCCESS)
        return nullptr;

    vulkan_check(vkBindImageMemory(dev, (VkImage) image, memory, 0));
    *size_out = (size_t) req.size;
    return (void *) memory;
}

void vulkan_buffer_free(void *buffer) {
    VulkanBuffer *b = (VulkanBuffer *) buffer;
    vkDestroyBuffer((VkDevice) b->device, (VkBuffer) b->buffer, nullptr);
    vkFreeMemory((VkDevice) b->device, (VkDeviceMemory) b->memory, nullptr);
    delete b;
}

/// Maps Dr.Jit pointers to the VulkanBuffer that contains them
static BufferMap vulkan_buffer_map;

void jitc_vulkan_register_buffer(void *ptr, void *buffer, size_t size) {
    // Device addresses and mapped host pointers share this map. Vulkan does
    // not guarantee that their address ranges are disjoint.
    size_t offset;
    if (unlikely(vulkan_buffer_map.find(ptr, &offset)))
        jitc_fail("jit_malloc(): the Vulkan allocation at " DRJIT_PTR
                  " overlaps an existing allocation!", (uintptr_t) ptr);
    vulkan_buffer_map.insert(ptr, buffer, size);
}

void *jitc_vulkan_find_buffer(void *ptr, size_t *offset_out) {
    return vulkan_buffer_map.find(ptr, offset_out);
}

void *jitc_vulkan_unregister_buffer(void *ptr) {
    return vulkan_buffer_map.remove(ptr);
}

uint64_t jitc_vulkan_device_address(void *ptr) {
    size_t offset = 0;
    VulkanBuffer *buf = (VulkanBuffer *) vulkan_buffer_map.find(ptr, &offset);
    if (unlikely(!buf))
        jitc_fail("jitc_vulkan_device_address(): " DRJIT_PTR " is not a "
                  "Vulkan allocation!", (uintptr_t) ptr);

    VkBufferDeviceAddressInfo address_info {};
    address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    address_info.buffer = (VkBuffer) buf->buffer;
    return vkGetBufferDeviceAddress((VkDevice) buf->device, &address_info) +
           offset;
}

// ============================================================================
//  Utility kernels
// ============================================================================

/// Entry point names, indexed by VulkanKernel. The order must match the
/// VulkanKernel enum in internal.h.
static const char *vulkan_kernel_names[(uint32_t) VulkanKernel::Count] = {
    "aggregate",
    "deinterleave_u8", "deinterleave_u16", "deinterleave_u32",
    "interleave_u8", "interleave_u16", "interleave_u32"
};

/// Create the pipelines of the utility kernels in the module ``spirv``. Their
/// workgroup size (specialization constant 0) is that of the generated kernels.
static void jitc_vulkan_create_utility_pipelines(VulkanDevice &device,
                                                 const char *spirv,
                                                 size_t size) {
    VkDevice dev = (VkDevice) device.device;

    VkShaderModuleCreateInfo module_info {};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = size;
    module_info.pCode = (const uint32_t *) spirv;
    vulkan_check(vkCreateShaderModule(
        dev, &module_info, nullptr, (VkShaderModule *) &device.utility_module));

    VkSpecializationMapEntry entry { 0, 0, sizeof(uint32_t) };
    VkSpecializationInfo spec {};
    spec.mapEntryCount = 1;
    spec.pMapEntries = &entry;
    spec.dataSize = sizeof(uint32_t);
    spec.pData = &device.block_size;

    for (uint32_t i = 0; i < (uint32_t) VulkanKernel::Count; ++i) {
        VkComputePipelineCreateInfo pipeline_info {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = (VkShaderModule) device.utility_module;
        pipeline_info.stage.pName = vulkan_kernel_names[i];
        pipeline_info.stage.pSpecializationInfo = &spec;
        pipeline_info.layout = (VkPipelineLayout) device.layout;
        vulkan_check(vkCreateComputePipelines(
            dev, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
            (VkPipeline *) &device.pipelines[i]));
    }
}

// ============================================================================
//  Completion threads
// ============================================================================

/**
 * Vulkan cannot notify the host when work completes. Each device therefore has
 * a thread that waits on the timeline semaphore and invokes host callbacks
 * once the timeline reaches their value, like the driver threads of CUDA
 * (cuLaunchHostFunc) and Metal (completion handlers). The callbacks run
 * without holding the Dr.Jit lock.
 *
 * Callbacks are queued with the device's most recent timeline value, hence the
 * queue is sorted and processed in order. An idle thread sleeps on the 'wake'
 * semaphore, which the host signals when a callback arrives or at shutdown.
 */
struct VulkanCompletion {
    struct Callback {
        uint64_t value;
        void (*callback)(void *);
        void *payload;
    };

    /// Protects the fields below
    Lock lock;

    /// Callbacks waiting for the timeline, sorted by value
    std::vector<Callback> pending;

    /// Host-signaled timeline semaphore that wakes the idle thread
    VkSemaphore wake = VK_NULL_HANDLE;
    uint64_t wake_value = 0;

    /// Set during shutdown
    bool stop = false;

    std::thread thread;
};

/// Completion thread state of each entry of ``state.vulkan_devices``
static std::vector<VulkanCompletion *> vulkan_completions;

static VkSemaphore jitc_vulkan_create_timeline(VkDevice device) {
    VkSemaphoreTypeCreateInfo type_info {};
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;

    VkSemaphoreCreateInfo semaphore_info {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_info.pNext = &type_info;

    VkSemaphore semaphore = VK_NULL_HANDLE;
    vulkan_check(vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore));
    return semaphore;
}

/// Wake the idle completion thread. Requires ``c->lock``, which keeps the
/// signaled values increasing.
static void jitc_vulkan_wake(VulkanCompletion *c, VkDevice device) {
    VkSemaphoreSignalInfo signal_info {};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal_info.semaphore = c->wake;
    signal_info.value = ++c->wake_value;
    vulkan_check(vkSignalSemaphore(device, &signal_info));
}

static void jitc_vulkan_completion_thread(VulkanCompletion *c, VkDevice device,
                                          VkSemaphore timeline) {
    while (true) {
        VulkanCompletion::Callback cb;
        uint64_t wake;
        bool idle;

        /* Fetch the next callback */ {
            lock_guard guard(c->lock);
            idle = c->pending.empty();
            if (idle && c->stop)
                break;
            if (!idle)
                cb = c->pending.front();
            wake = c->wake_value + 1;
        }

        if (idle) {
            vulkan_wait(device, c->wake, wake);
            continue;
        }

        vulkan_wait(device, timeline, cb.value);
        cb.callback(cb.payload);

        lock_guard guard(c->lock);
        c->pending.erase(c->pending.begin());
    }
}

void jitc_vulkan_enqueue_callback(int device, void (*callback)(void *),
                                  void *payload) {
    VulkanCompletion *c = vulkan_completions[device];
    const VulkanDevice &dev = state.vulkan_devices[device];

    lock_guard guard(c->lock);
    c->pending.push_back({ dev.submitted, callback, payload });

    if (c->pending.size() == 1)
        jitc_vulkan_wake(c, (VkDevice) dev.device);
}

// ============================================================================
//  Kernel compilation
// ============================================================================

/// Vulkan unit artifacts store a VkShaderModule in ptr[0] and its
/// VkDevice in ptr[1]. The device index serves as the cache salt.
static void jitc_vulkan_unit_release(UnitArtifact &a) {
    vkDestroyShaderModule((VkDevice) a.ptr[1], (VkShaderModule) a.ptr[0],
                          nullptr);
}

/// Fold device/driver identity and cache version into a disk-cache key
static XXH128_hash_t jitc_vulkan_disk_key(const VulkanDevice &dev,
                                          XXH128_hash_t hash) {
    // A seeded reset consults the previous state, so it must be initialized
    XXH3_state_t xs;
    XXH3_INITSTATE(&xs);
    XXH3_128bits_reset_withSeed(&xs, DRJIT_CACHE_VERSION);
    XXH3_128bits_update(&xs, &hash, sizeof(XXH128_hash_t));
    XXH3_128bits_update(&xs, dev.pipeline_cache_uuid,
                        sizeof(dev.pipeline_cache_uuid));
    return XXH3_128bits_digest(&xs);
}

/// Append the body of ``unit`` to ``out`` and replace the placeholders
/// '@<hash>' of call dispatches by callable indices. Decorations go to
/// ``annotations`` instead, since SPIR-V requires them to precede all types. A
/// non-null ``prefix`` is prepended to the IDs defined in the body.
static void jitc_vulkan_link_body(StringBuffer &out, StringBuffer &annotations,
                                  const UnitBuilder *unit, const char *prefix) {
    const char *start = unit->src.get() + unit->body_offset,
               *end = unit->src.get() + unit->src.size();
    auto is_id_char = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_';
    };

    // IDs defined by the body: '%<name> = ...' (except for function names)
    tsl::robin_set<std::string_view> defined;
    if (prefix) {
        for (const char *p = start; p < end; ) {
            const char *eol = (const char *) memchr(p, '\n', end - p);
            if (!eol)
                eol = end;
            if (*p == '%' && strncmp(p + 1, "func_", 5) != 0) {
                const char *q = p + 1;
                while (q < eol && is_id_char(*q))
                    q++;
                if (strncmp(q, " = ", 3) == 0)
                    defined.insert(std::string_view(p + 1, q - p - 1));
            }
            p = eol + 1;
        }
    }

    StringBuffer *dst = &out;
    for (const char *p = start; p < end; ) {
        if (p == start || p[-1] == '\n')
            dst = strncmp(p, "OpDecorate ", 11) == 0 ? &annotations : &out;

        if (*p == '%' && prefix) {
            const char *q = p + 1;
            while (q < end && is_id_char(*q))
                q++;
            dst->put('%');
            if (defined.count(std::string_view(p + 1, q - p - 1)))
                dst->put(prefix, strlen(prefix));
            dst->put(p + 1, q - p - 1);
            p = q;
        } else if (*p == '@') {
            char tmp[17] = { };
            XXH128_hash_t hash;
            memcpy(tmp, p + 1, 16);
            hash.high64 = strtoull(tmp, nullptr, 16);
            memcpy(tmp, p + 17, 16);
            hash.low64 = strtoull(tmp, nullptr, 16);
            dst->put_u32(jitc_unit_callable_index(hash));
            p += 33;
        } else {
            dst->put(*p++);
        }
    }
}

/// Source of the module combining the units of the kernel being compiled, and
/// the annotations and functions collected while building it
static StringBuffer jitc_vulkan_module_src { 1024 };
static StringBuffer jitc_vulkan_module_annotations { 1024 };
static StringBuffer jitc_vulkan_module_funcs { 1024 };

/// Combine the units of a kernel into a single SPIR-V module, the counterpart
/// of the link step of the other backends. The module consists of the
/// prologue, the decorations of the functions, the declarations of all units
/// (without duplicates), and the functions. IDs are global in a SPIR-V
/// module, hence those defined by callable 'k' are renamed to '%f<k>_<name>'.
static void jitc_vulkan_link(StringBuffer &out) {
    StringBuffer &annotations = jitc_vulkan_module_annotations,
                 &funcs = jitc_vulkan_module_funcs;
    annotations.clear();
    funcs.clear();

    jitc_vulkan_link_body(funcs, annotations, unit_entry, nullptr);
    for (size_t k = 0; k < callable_units.size(); ++k) {
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "f%zu_", k);
        jitc_vulkan_link_body(funcs, annotations, callable_units[k].unit,
                              prefix);
    }

    // The prologue ends with the annotation section
    out.clear();
    out.put(unit_entry->src.get(), unit_prologue.size());
    out.put(annotations.get(), annotations.size());
    out.put(unit_entry->src.get() + unit_prologue.size(),
            unit_entry->body_offset - unit_prologue.size());

    tsl::robin_set<XXH128_hash_t, XXH128Hasher, XXH128Eq> declared =
        unit_entry->registered;
    for (const CallableUnit &cu : callable_units) {
        const char *p = cu.unit->src.get() + unit_prologue.size(),
                   *end = cu.unit->src.get() + cu.unit->body_offset;
        while (p < end) {
            const char *eol = (const char *) memchr(p, '\n', end - p);
            if (declared.insert(XXH128(p, eol - p, 0)).second)
                out.put(p, eol - p + 1);
            p = eol + 1;
        }
    }

    out.put(funcs.get(), funcs.size());
}

bool jitc_vulkan_kernel_compile(ThreadState *ts, Kernel &kernel) {
    const VulkanDevice &dev = state.vulkan_devices[ts->device];
    VkDevice device = (VkDevice) dev.device;

    UnitCompileJob job;
    jitc_unit_job_init(0, job);
    jitc_vulkan_link(jitc_vulkan_module_src);
    job.source = jitc_vulkan_module_src.get();
    job.source_size = jitc_vulkan_module_src.size();

    UnitArtifact artifact;
    // The module is identified by the kernel hash, which also covers the
    // callables
    bool hit = jitc_unit_cache_lookup(JitBackend::Vulkan, kernel_hash,
                                      (uint64_t) ts->device, artifact);
    VkShaderModule module = hit ? (VkShaderModule) artifact.ptr[0]
                                : VK_NULL_HANDLE;

    XXH128_hash_t key = jitc_vulkan_disk_key(dev, kernel_hash);

    VkPipeline pipeline = VK_NULL_HANDLE;
    bool module_from_disk = false, pipeline_from_disk = false;

    // Release the lock while compiling. The job source stays valid throughout
    // (see UnitCompileJob in unit.h)
    {
        unlock_guard guard(state.lock);

        if (!hit) {
            // SPIR-V binary from the disk cache, or from the assembler
            std::vector<uint8_t> spirv;
            module_from_disk = jitc_cache_blob_load(CacheKind::VulkanModule,
                                                    key, spirv);
            if (!module_from_disk) {
                std::string error;
                size_t line = 0;
                if (!vulkan_asm(job.source, job.source_size, spirv, error,
                                line))
                    jitc_fail("jitc_vulkan_kernel_compile(): assembly of "
                              "\"%s\" failed in line %zu: %s\n\n--- Source "
                              "code ---\n%s", job.symbol, line + 1,
                              error.c_str(), job.source);

                if (jitc_cache_writable())
                    jitc_cache_blob_store_async(CacheKind::VulkanModule,
                                                key,
                                                std::vector<uint8_t>(spirv));
            }

            // Drivers may crash on invalid modules, hence validate them in
            // debug mode (if SPIRV-Tools is available)
            if (jit_flag(JitFlag::Debug) && jitc_vulkan_spirv_context) {
                spv_diagnostic diagnostic = nullptr;
                if (spvValidateBinary(jitc_vulkan_spirv_context,
                                      (const uint32_t *) spirv.data(),
                                      spirv.size() / sizeof(uint32_t),
                                      &diagnostic) != SPV_SUCCESS)
                    jitc_fail("jitc_vulkan_kernel_compile(): validation of "
                              "\"%s\" failed: %s\n\n--- Source code ---\n%s",
                              job.symbol,
                              diagnostic ? diagnostic->error : "<unknown>",
                              job.source);
                spvDiagnosticDestroy(diagnostic);
            }

            VkShaderModuleCreateInfo module_info {};
            module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            module_info.codeSize = spirv.size();
            module_info.pCode = (const uint32_t *) spirv.data();
            vulkan_check(vkCreateShaderModule(device, &module_info, nullptr,
                                              &module));
        }

        // The pipeline cache persists the machine code produced by the driver
        std::vector<uint8_t> cache_data;
        pipeline_from_disk = jitc_cache_blob_load(CacheKind::VulkanPipeline,
                                                  key, cache_data);

        VkPipelineCacheCreateInfo cache_info {};
        cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        cache_info.initialDataSize = cache_data.size();
        cache_info.pInitialData = cache_data.data();

        VkPipelineCache cache = VK_NULL_HANDLE;
        vulkan_check(vkCreatePipelineCache(device, &cache_info, nullptr, &cache));

        VkComputePipelineCreateInfo pipeline_info {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = job.symbol;
        pipeline_info.layout = (VkPipelineLayout) dev.layout;
        vulkan_check(vkCreateComputePipelines(device, cache, 1, &pipeline_info,
                                              nullptr, &pipeline));

        if (!pipeline_from_disk && jitc_cache_writable()) {
            size_t size = 0;
            vulkan_check(vkGetPipelineCacheData(device, cache, &size, nullptr));
            std::vector<uint8_t> data(size);
            vulkan_check(
                vkGetPipelineCacheData(device, cache, &size, data.data()));
            jitc_cache_blob_store_async(CacheKind::VulkanPipeline,
                                        key, std::move(data));
        }

        vkDestroyPipelineCache(device, cache, nullptr);
    }

    // Publish the new artifacts; the cache pins the shader module
    if (!hit) {
        UnitArtifact a { { (void *) module, (void *) device }, 0, 0 };
        jitc_unit_cache_insert(JitBackend::Vulkan, kernel_hash,
                               (uint64_t) ts->device, a,
                               jitc_vulkan_unit_release);
    }

    kernel.vulkan.pipeline = (void *) pipeline;
    kernel.size = (uint32_t) job.source_size;

    // Report a soft miss when the module and pipeline were reconstructed from
    // the in-memory or disk caches
    return (hit || module_from_disk) && pipeline_from_disk;
}

void jitc_vulkan_kernel_free(int device, Kernel &kernel) {
    vkDestroyPipeline((VkDevice) state.vulkan_devices[device].device,
                      (VkPipeline) kernel.vulkan.pipeline, nullptr);
}

// ============================================================================
//  Backend init / shutdown
// ============================================================================

/// Choose the number of threads per workgroup of the kernels generated for
/// ``pdev``. Vulkan provides no occupancy information, hence this is a
/// heuristic: one subgroup for each of the (typically four) SIMD partitions of
/// an SM (NVIDIA) or compute unit (AMD).
static uint32_t jitc_vulkan_block_size(VkPhysicalDevice pdev) {
    VkPhysicalDeviceVulkan11Properties p11 {};
    p11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;

    VkPhysicalDeviceProperties2 p {};
    p.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p.pNext = &p11;
    vkGetPhysicalDeviceProperties2(pdev, &p);

    return std::min(4 * p11.subgroupSize,
                    p.properties.limits.maxComputeWorkGroupInvocations);
}

/// Number of textures per dimension of the texture heap of ``pdev``. Every
/// texture occupies a sampled image and a sampler descriptor.
static uint32_t jitc_vulkan_heap_size(VkPhysicalDevice pdev) {
    VkPhysicalDeviceVulkan12Properties p12 {};
    p12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;

    VkPhysicalDeviceProperties2 p {};
    p.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p.pNext = &p12;
    vkGetPhysicalDeviceProperties2(pdev, &p);

    uint32_t limit = std::min({
        p12.maxPerStageDescriptorUpdateAfterBindSampledImages,
        p12.maxPerStageDescriptorUpdateAfterBindSamplers,
        p12.maxDescriptorSetUpdateAfterBindSampledImages,
        p12.maxDescriptorSetUpdateAfterBindSamplers,
        p12.maxPerStageUpdateAfterBindResources });
    return std::min(limit / 3, 65536u);
}

/// Return the name of the first feature required by Dr.Jit that ``pdev``
/// lacks, or nullptr if it supports all of them
static const char *jitc_vulkan_missing_feature(VkPhysicalDevice pdev) {
    VkPhysicalDeviceVulkan12Features f12 {};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceVulkan11Features f11 {};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f11.pNext = &f12;

    VkPhysicalDeviceFeatures2 f {};
    f.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f.pNext = &f11;
    vkGetPhysicalDeviceFeatures2(pdev, &f);

    if (!f.features.shaderInt16)         return "shaderInt16";
    if (!f.features.shaderInt64)         return "shaderInt64";
    if (!f.features.shaderFloat64)       return "shaderFloat64";
    if (!f12.shaderInt8)                 return "shaderInt8";
    if (!f12.shaderBufferInt64Atomics)   return "shaderBufferInt64Atomics";
    if (!f12.shaderFloat16)              return "shaderFloat16";
    if (!f11.storageBuffer16BitAccess)   return "storageBuffer16BitAccess";
    if (!f12.storageBuffer8BitAccess)    return "storageBuffer8BitAccess";
    if (!f12.bufferDeviceAddress)        return "bufferDeviceAddress";
    if (!f12.timelineSemaphore)          return "timelineSemaphore";

    // Texture heap (see vulkan_tex.cpp)
    if (!f12.runtimeDescriptorArray)     return "runtimeDescriptorArray";
    if (!f12.descriptorBindingPartiallyBound)
        return "descriptorBindingPartiallyBound";
    if (!f12.descriptorBindingSampledImageUpdateAfterBind)
        return "descriptorBindingSampledImageUpdateAfterBind";
    if (!f12.descriptorBindingUpdateUnusedWhilePending)
        return "descriptorBindingUpdateUnusedWhilePending";
    if (!f12.shaderSampledImageArrayNonUniformIndexing)
        return "shaderSampledImageArrayNonUniformIndexing";
    if (!f.features.samplerAnisotropy)   return "samplerAnisotropy";
    return nullptr;
}

bool jitc_vulkan_init() {
    /// Was the Vulkan backend already initialized?
    if (jitc_vulkan_instance)
        return true;

    // First, dynamically load the Vulkan loader into the process
    if (!jitc_vulkan_api_init())
        return false;

    uint32_t version = 0;
    vulkan_check(vkEnumerateInstanceVersion(&version));

    if (version < VK_API_VERSION_1_2) {
        jitc_vulkan_api_shutdown();
        jitc_log(Warn,
                 "jit_vulkan_init(): your version of Vulkan is too old (found "
                 "%u.%u, at least 1.2 is required) -- disabling Vulkan backend!",
                 VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version));
        return false;
    }

    VkApplicationInfo app_info {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pEngineName = "Dr.Jit";
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    // The following call may fail if no Vulkan driver is installed
    VkResult rv =
        vkCreateInstance(&instance_info, nullptr, &jitc_vulkan_instance);
    if (rv != VK_SUCCESS) {
        jitc_vulkan_instance = nullptr;
        jitc_vulkan_api_shutdown();
        jitc_log(Warn,
                 "jit_vulkan_init(): vkCreateInstance() failed (error %i) -- "
                 "disabling Vulkan backend!", (int) rv);
        return false;
    }

    if (!jitc_vulkan_api_init_instance(jitc_vulkan_instance)) {
        jitc_vulkan_shutdown();
        return false;
    }

    if (spvContextCreate)
        jitc_vulkan_spirv_context = spvContextCreate(SPV_ENV_VULKAN_1_2);
    else
        jitc_log(LogLevel::Info, "jit_vulkan_init(): SPIRV-Tools not found, "
                 "kernels will not be validated in debug mode.");

    uint32_t device_count = 0;
    vulkan_check(vkEnumeratePhysicalDevices(jitc_vulkan_instance,
                                            &device_count, nullptr));
    if (device_count == 0) {
        jitc_vulkan_shutdown();
        jitc_log(Warn, "jit_vulkan_init(): No devices found -- disabling "
                       "Vulkan backend!");
        return false;
    }

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    vulkan_check(vkEnumeratePhysicalDevices(jitc_vulkan_instance,
                                            &device_count,
                                            physical_devices.data()));

    jitc_log(Info, "jit_vulkan_init(): enabling Vulkan backend (version %u.%u)",
             VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version));

    for (uint32_t i = 0; i < device_count; ++i) {
        VkPhysicalDevice pdev = physical_devices[i];

        VkPhysicalDeviceProperties2 props {};
        props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        vkGetPhysicalDeviceProperties2(pdev, &props);
        const VkPhysicalDeviceProperties &p = props.properties;

        uint32_t block_size = jitc_vulkan_block_size(pdev);
        jitc_log(Info,
                 " - Found Vulkan device %u: \"%s\" (Vulkan %u.%u, %u "
                 "threads/workgroup)", i, p.deviceName,
                 VK_API_VERSION_MAJOR(p.apiVersion),
                 VK_API_VERSION_MINOR(p.apiVersion), block_size);

        if (p.apiVersion < VK_API_VERSION_1_2) {
            jitc_log(Warn, " - Warning: device does *not* support Vulkan 1.2, "
                           "skipping ..");
            continue;
        }

        const char *missing = jitc_vulkan_missing_feature(pdev);
        if (missing) {
            jitc_log(Warn, " - Warning: device does *not* support \"%s\", "
                           "skipping ..", missing);
            continue;
        }

        // Find a queue family that supports compute work
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count,
                                                 families.data());

        uint32_t family = (uint32_t) -1;
        for (uint32_t j = 0; j < family_count; ++j) {
            if (families[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                family = j;
                break;
            }
        }

        if (family == (uint32_t) -1) {
            jitc_log(Warn, " - Warning: device has no compute queue, "
                           "skipping ..");
            continue;
        }

        float priority = 1.f;
        VkDeviceQueueCreateInfo queue_info {};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;

        // Enable the features checked by jitc_vulkan_missing_feature()
        VkPhysicalDeviceVulkan12Features f12 {};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        f12.shaderInt8 = VK_TRUE;
        f12.shaderBufferInt64Atomics = VK_TRUE;
        f12.shaderFloat16 = VK_TRUE;
        f12.storageBuffer8BitAccess = VK_TRUE;
        f12.bufferDeviceAddress = VK_TRUE;
        f12.timelineSemaphore = VK_TRUE;
        f12.runtimeDescriptorArray = VK_TRUE;
        f12.descriptorBindingPartiallyBound = VK_TRUE;
        f12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
        f12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
        f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

        VkPhysicalDeviceVulkan11Features f11 {};
        f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        f11.pNext = &f12;
        f11.storageBuffer16BitAccess = VK_TRUE;

        // Optional: atomic floating point addition. The queried features of
        // the extension are enabled as a whole.
        uint32_t n_ext = 0;
        vkEnumerateDeviceExtensionProperties(pdev, nullptr, &n_ext, nullptr);
        std::vector<VkExtensionProperties> ext(n_ext);
        vkEnumerateDeviceExtensionProperties(pdev, nullptr, &n_ext, ext.data());

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float {};
        atomic_float.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        const char *extensions[1];
        uint32_t n_extensions = 0;
        for (const VkExtensionProperties &e : ext) {
            if (strcmp(e.extensionName,
                       VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) != 0)
                continue;
            VkPhysicalDeviceFeatures2 query {};
            query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            query.pNext = &atomic_float;
            vkGetPhysicalDeviceFeatures2(pdev, &query);
            f12.pNext = &atomic_float;
            extensions[n_extensions++] = VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME;
        }

        VkPhysicalDeviceFeatures2 features {};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &f11;
        features.features.shaderInt16 = VK_TRUE;
        features.features.shaderInt64 = VK_TRUE;
        features.features.shaderFloat64 = VK_TRUE;
        features.features.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo device_info {};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.pNext = &features;
        device_info.enabledExtensionCount = n_extensions;
        device_info.ppEnabledExtensionNames = extensions;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;

        VkDevice dev = nullptr;
        rv = vkCreateDevice(pdev, &device_info, nullptr, &dev);
        if (rv != VK_SUCCESS) {
            jitc_log(Warn, " - Warning: vkCreateDevice() failed (error %i), "
                           "skipping ..", (int) rv);
            continue;
        }

        VulkanDevice device {};
        device.physical_device = pdev;
        device.device = dev;
        device.queue_family = family;
        std::memcpy(device.pipeline_cache_uuid, p.pipelineCacheUUID,
                    VK_UUID_SIZE);
        vkGetDeviceQueue(dev, family, 0, (VkQueue *) &device.queue);

        VkCommandPoolCreateInfo pool_info {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = family;
        vulkan_check(vkCreateCommandPool(dev, &pool_info, nullptr,
                                         (VkCommandPool *) &device.pool));

        device.timeline = jitc_vulkan_create_timeline(dev);

        // The minimum size of push constants that every device supports
        VkPushConstantRange push_range {};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.size = 128;

        jitc_vulkan_heap_init(device, jitc_vulkan_heap_size(pdev));

        VkPipelineLayoutCreateInfo layout_info {};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = (VkDescriptorSetLayout *) &device.heap_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_range;
        vulkan_check(vkCreatePipelineLayout(dev, &layout_info, nullptr,
                                            (VkPipelineLayout *) &device.layout));

        device.max_group_count = p.limits.maxComputeWorkGroupCount[0];
        device.block_size = block_size;
        device.atomic_float32_add = atomic_float.shaderBufferFloat32AtomicAdd;
        device.atomic_float64_add = atomic_float.shaderBufferFloat64AtomicAdd;

        char *spirv = jitc_lz4_inflate(vulkan_kernels,
                                       vulkan_kernels_size_compressed,
                                       vulkan_kernels_size_uncompressed,
                                       "utility kernel module");
        jitc_vulkan_create_utility_pipelines(device, spirv,
                                             vulkan_kernels_size_uncompressed);
        free(spirv);

        size_t len = std::strlen(p.deviceName);
        device.name = (char *) std::malloc(len + 1);
        std::memcpy(device.name, p.deviceName, len + 1);

        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(pdev, &memory_properties);

        VulkanCompletion *completion = new VulkanCompletion();
        lock_init(completion->lock);
        completion->wake = jitc_vulkan_create_timeline(dev);
        completion->thread =
            std::thread(jitc_vulkan_completion_thread, completion, dev,
                        (VkSemaphore) device.timeline);

        state.vulkan_devices.push_back(device);
        vulkan_memory_properties.push_back(memory_properties);
        vulkan_completions.push_back(completion);
    }

    return !state.vulkan_devices.empty();
}

void vulkan_wait(void *device, void *timeline, uint64_t value) {
    VkSemaphore semaphore = (VkSemaphore) timeline;

    VkSemaphoreWaitInfo wait_info {};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &semaphore;
    wait_info.pValues = &value;
    vulkan_check(vkWaitSemaphores((VkDevice) device, &wait_info, UINT64_MAX));
}

void jitc_vulkan_sync(ThreadState *ts) {
    VulkanThreadState *vts = (VulkanThreadState *) ts->actual_state();
    vts->flush(/* wait = */ false);

    // Release the lock while waiting
    unlock_guard guard(state.lock);
    vulkan_wait(vts->vulkan_device, vts->vulkan_timeline, vts->vulkan_last);
}

void jitc_vulkan_flush(ThreadState *ts) {
    ((VulkanThreadState *) ts->actual_state())->flush(/* wait = */ false);
}

void jitc_vulkan_sync_devices() {
    std::vector<VulkanDevice> devices = state.vulkan_devices;

    // Release the lock while waiting
    unlock_guard guard(state.lock);
    for (const VulkanDevice &dev : devices)
        vulkan_wait(dev.device, dev.timeline, dev.submitted);
}

void jitc_vulkan_shutdown() {
    jitc_log(Info, "jit_vulkan_shutdown()");

    jitc_unit_cache_flush((int) JitBackend::Vulkan);

    for (size_t i = 0; i < state.vulkan_devices.size(); ++i) {
        VulkanDevice &dev = state.vulkan_devices[i];
        VulkanCompletion *c = vulkan_completions[i];

        /* Run the remaining callbacks, then stop the completion thread */ {
            lock_guard guard(c->lock);
            c->stop = true;
            jitc_vulkan_wake(c, (VkDevice) dev.device);
        }

        /* Callbacks may acquire the Dr.Jit lock */ {
            unlock_guard guard(state.lock);
            c->thread.join();
        }
        vkDestroySemaphore((VkDevice) dev.device, c->wake, nullptr);
        lock_destroy(c->lock);
        delete c;

        vulkan_check(vkDeviceWaitIdle((VkDevice) dev.device));
        vkDestroySemaphore((VkDevice) dev.device, (VkSemaphore) dev.timeline,
                           nullptr);
        vkDestroyCommandPool((VkDevice) dev.device, (VkCommandPool) dev.pool,
                             nullptr);
        for (uint32_t i = 0; i < (uint32_t) VulkanKernel::Count; ++i)
            vkDestroyPipeline((VkDevice) dev.device,
                              (VkPipeline) dev.pipelines[i], nullptr);
        vkDestroyShaderModule((VkDevice) dev.device,
                              (VkShaderModule) dev.utility_module, nullptr);
        vkDestroyPipelineLayout((VkDevice) dev.device,
                                (VkPipelineLayout) dev.layout, nullptr);
        jitc_vulkan_heap_shutdown(dev);
        vkDestroyDevice((VkDevice) dev.device, nullptr);
        std::free(dev.name);
    }
    state.vulkan_devices.clear();
    vulkan_memory_properties.clear();
    vulkan_completions.clear();
    vulkan_buffer_map.clear();

    if (jitc_vulkan_spirv_context) {
        spvContextDestroy(jitc_vulkan_spirv_context);
        jitc_vulkan_spirv_context = nullptr;
    }

    if (jitc_vulkan_instance) {
        if (vkDestroyInstance)
            vkDestroyInstance(jitc_vulkan_instance, nullptr);
        jitc_vulkan_instance = nullptr;
    }

    jitc_vulkan_api_shutdown();
}

#endif // defined(DRJIT_ENABLE_VULKAN)
