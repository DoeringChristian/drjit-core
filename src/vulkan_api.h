/*
    src/vulkan_api.h -- Low-level interface to the Vulkan API

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>

/// Try to load the Vulkan loader and resolve its global API functions
extern bool jitc_vulkan_api_init();

/// Resolve the API functions that are dispatched through ``instance``
extern bool jitc_vulkan_api_init_instance(VkInstance instance);

/// Free any resources allocated by jitc_vulkan_api_init()
extern void jitc_vulkan_api_shutdown();

#define vulkan_check(err) vulkan_check_impl(err, __FILE__, __LINE__)
extern void vulkan_check_impl(VkResult errval, const char *file, const int line);

#if !defined(DR_VULKAN_SYM)
#  define DR_VULKAN_SYM(x) extern x;
#endif

// Global functions
DR_VULKAN_SYM(PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr);
DR_VULKAN_SYM(PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion);
DR_VULKAN_SYM(PFN_vkCreateInstance vkCreateInstance);

// Instance functions
DR_VULKAN_SYM(PFN_vkDestroyInstance vkDestroyInstance);
DR_VULKAN_SYM(PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices);
DR_VULKAN_SYM(PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2);
DR_VULKAN_SYM(PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2);
DR_VULKAN_SYM(PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties);
DR_VULKAN_SYM(PFN_vkCreateDevice vkCreateDevice);

// Device functions (resolved through the instance)
DR_VULKAN_SYM(PFN_vkDestroyDevice vkDestroyDevice);
DR_VULKAN_SYM(PFN_vkGetDeviceQueue vkGetDeviceQueue);
DR_VULKAN_SYM(PFN_vkDeviceWaitIdle vkDeviceWaitIdle);
