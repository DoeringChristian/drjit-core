/*
    src/vulkan_core.cpp -- Vulkan device init and shutdown

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(DRJIT_ENABLE_VULKAN)

#include "vulkan.h"
#include "vulkan_api.h"
#include "vulkan_ts.h"
#include "buffer_map.h"
#include "internal.h"
#include "log.h"

#include <cstring>
#include <cstdlib>
#include <vector>
#include <thread>

static VkInstance jitc_vulkan_instance = nullptr;

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
//  Backend init / shutdown
// ============================================================================

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

    if (!f.features.shaderInt64)         return "shaderInt64";
    if (!f11.storageBuffer16BitAccess)   return "storageBuffer16BitAccess";
    if (!f12.storageBuffer8BitAccess)    return "storageBuffer8BitAccess";
    if (!f12.bufferDeviceAddress)        return "bufferDeviceAddress";
    if (!f12.timelineSemaphore)          return "timelineSemaphore";
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

        jitc_log(Info, " - Found Vulkan device %u: \"%s\" (Vulkan %u.%u)", i,
                 p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion),
                 VK_API_VERSION_MINOR(p.apiVersion));

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
        f12.storageBuffer8BitAccess = VK_TRUE;
        f12.bufferDeviceAddress = VK_TRUE;
        f12.timelineSemaphore = VK_TRUE;

        VkPhysicalDeviceVulkan11Features f11 {};
        f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        f11.pNext = &f12;
        f11.storageBuffer16BitAccess = VK_TRUE;

        VkPhysicalDeviceFeatures2 features {};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &f11;
        features.features.shaderInt64 = VK_TRUE;

        VkDeviceCreateInfo device_info {};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.pNext = &features;
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
        vkGetDeviceQueue(dev, family, 0, (VkQueue *) &device.queue);

        VkCommandPoolCreateInfo pool_info {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = family;
        vulkan_check(vkCreateCommandPool(dev, &pool_info, nullptr,
                                         (VkCommandPool *) &device.pool));

        device.timeline = jitc_vulkan_create_timeline(dev);

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
        vkDestroyDevice((VkDevice) dev.device, nullptr);
        std::free(dev.name);
    }
    state.vulkan_devices.clear();
    vulkan_memory_properties.clear();
    vulkan_completions.clear();
    vulkan_buffer_map.clear();

    if (jitc_vulkan_instance) {
        if (vkDestroyInstance)
            vkDestroyInstance(jitc_vulkan_instance, nullptr);
        jitc_vulkan_instance = nullptr;
    }

    jitc_vulkan_api_shutdown();
}

#endif // defined(DRJIT_ENABLE_VULKAN)
