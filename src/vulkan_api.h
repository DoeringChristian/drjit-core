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
DR_VULKAN_SYM(PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties);
DR_VULKAN_SYM(PFN_vkCreateDevice vkCreateDevice);

// Device functions (resolved through the instance)
DR_VULKAN_SYM(PFN_vkDestroyDevice vkDestroyDevice);
DR_VULKAN_SYM(PFN_vkGetDeviceQueue vkGetDeviceQueue);
DR_VULKAN_SYM(PFN_vkDeviceWaitIdle vkDeviceWaitIdle);
DR_VULKAN_SYM(PFN_vkCreateBuffer vkCreateBuffer);
DR_VULKAN_SYM(PFN_vkDestroyBuffer vkDestroyBuffer);
DR_VULKAN_SYM(PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements);
DR_VULKAN_SYM(PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress);
DR_VULKAN_SYM(PFN_vkAllocateMemory vkAllocateMemory);
DR_VULKAN_SYM(PFN_vkFreeMemory vkFreeMemory);
DR_VULKAN_SYM(PFN_vkBindBufferMemory vkBindBufferMemory);
DR_VULKAN_SYM(PFN_vkMapMemory vkMapMemory);
DR_VULKAN_SYM(PFN_vkCreateCommandPool vkCreateCommandPool);
DR_VULKAN_SYM(PFN_vkDestroyCommandPool vkDestroyCommandPool);
DR_VULKAN_SYM(PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers);
DR_VULKAN_SYM(PFN_vkFreeCommandBuffers vkFreeCommandBuffers);
DR_VULKAN_SYM(PFN_vkBeginCommandBuffer vkBeginCommandBuffer);
DR_VULKAN_SYM(PFN_vkEndCommandBuffer vkEndCommandBuffer);
DR_VULKAN_SYM(PFN_vkQueueSubmit vkQueueSubmit);
DR_VULKAN_SYM(PFN_vkCreateSemaphore vkCreateSemaphore);
DR_VULKAN_SYM(PFN_vkDestroySemaphore vkDestroySemaphore);
DR_VULKAN_SYM(PFN_vkGetSemaphoreCounterValue vkGetSemaphoreCounterValue);
DR_VULKAN_SYM(PFN_vkWaitSemaphores vkWaitSemaphores);
DR_VULKAN_SYM(PFN_vkSignalSemaphore vkSignalSemaphore);
DR_VULKAN_SYM(PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier);
DR_VULKAN_SYM(PFN_vkCmdCopyBuffer vkCmdCopyBuffer);
DR_VULKAN_SYM(PFN_vkCmdFillBuffer vkCmdFillBuffer);
