/*
    src/vulkan_core.cpp -- Vulkan device init and shutdown

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(DRJIT_ENABLE_VULKAN)

#include "vulkan.h"
#include "vulkan_api.h"
#include "internal.h"
#include "log.h"

#include <cstring>
#include <cstdlib>
#include <vector>

static VkInstance jitc_vulkan_instance = nullptr;

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

        size_t len = std::strlen(p.deviceName);
        device.name = (char *) std::malloc(len + 1);
        std::memcpy(device.name, p.deviceName, len + 1);

        state.vulkan_devices.push_back(device);
    }

    return !state.vulkan_devices.empty();
}

void jitc_vulkan_shutdown() {
    jitc_log(Info, "jit_vulkan_shutdown()");

    for (VulkanDevice &dev : state.vulkan_devices) {
        vulkan_check(vkDeviceWaitIdle((VkDevice) dev.device));
        vkDestroyDevice((VkDevice) dev.device, nullptr);
        std::free(dev.name);
    }
    state.vulkan_devices.clear();

    if (jitc_vulkan_instance) {
        if (vkDestroyInstance)
            vkDestroyInstance(jitc_vulkan_instance, nullptr);
        jitc_vulkan_instance = nullptr;
    }

    jitc_vulkan_api_shutdown();
}

#endif // defined(DRJIT_ENABLE_VULKAN)
