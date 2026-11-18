/*
    src/vulkan_api.cpp -- Low-level interface to the Vulkan API

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#define DR_VULKAN_SYM(...) __VA_ARGS__ = nullptr;

#include "vulkan_api.h"
#include "log.h"
#include "internal.h"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

static void *jitc_vulkan_handle = nullptr;
static void *jitc_spirv_tools_handle = nullptr;

bool jitc_vulkan_api_init() {
    if (jitc_vulkan_handle)
        return true;

#if defined(_WIN32)
    const char *vulkan_fname = "vulkan-1.dll",
               *vulkan_glob  = nullptr,
               *spirv_fname  = "SPIRV-Tools-shared.dll",
               *spirv_glob   = nullptr;
#else
    const char *vulkan_fname = "libvulkan.so.1",
               *vulkan_glob  = "/usr/lib/{x86_64-linux-gnu,aarch64-linux-gnu}/libvulkan.so.*",
               *spirv_fname  = "libSPIRV-Tools-shared.so",
               *spirv_glob   = "/usr/lib/{x86_64-linux-gnu,aarch64-linux-gnu}/libSPIRV-Tools-shared.so*";
#endif

    jitc_vulkan_handle =
        jitc_find_library(vulkan_fname, vulkan_glob, "DRJIT_LIBVULKAN_PATH");
    if (!jitc_vulkan_handle) // Vulkan loader cannot be found, give up
        return false;

    // The Vulkan backend assembles kernels using SPIRV-Tools
    jitc_spirv_tools_handle = jitc_find_library(spirv_fname, spirv_glob,
                                                "DRJIT_LIBSPIRV_TOOLS_PATH");
    if (!jitc_spirv_tools_handle) {
        jitc_vulkan_api_shutdown();
        jitc_log(LogLevel::Warn,
                 "jit_vulkan_api_init(): could not find the SPIRV-Tools "
                 "library (\"%s\") -- disabling Vulkan backend! Set the "
                 "DRJIT_LIBSPIRV_TOOLS_PATH environment variable to specify "
                 "its path.", spirv_fname);
        return false;
    }

    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)
        dlsym(jitc_vulkan_handle, "vkGetInstanceProcAddr");

    const char *symbol = nullptr;

    do {
        if (!vkGetInstanceProcAddr) {
            symbol = "vkGetInstanceProcAddr";
            break;
        }

        #define LOAD(name)                                                   \
            symbol = #name;                                                  \
            name = (PFN_##name) vkGetInstanceProcAddr(nullptr, #name);       \
            if (!name)                                                       \
                break;                                                       \
            symbol = nullptr

        LOAD(vkEnumerateInstanceVersion);
        LOAD(vkCreateInstance);

        #undef LOAD

        #define LOAD(name)                                                   \
            symbol = #name;                                                  \
            name = decltype(name)(dlsym(jitc_spirv_tools_handle, #name));    \
            if (!name)                                                       \
                break;                                                       \
            symbol = nullptr

        LOAD(spvContextCreate);
        LOAD(spvContextDestroy);
        LOAD(spvTextToBinary);
        LOAD(spvBinaryDestroy);
        LOAD(spvDiagnosticDestroy);
        LOAD(spvValidateBinary);

        #undef LOAD
    } while (false);

    if (symbol) {
        jitc_vulkan_api_shutdown();
        jitc_log(LogLevel::Warn,
                 "jit_vulkan_api_init(): could not find symbol \"%s\" -- "
                 "disabling Vulkan backend!", symbol);
        return false;
    }

    return true;
}

bool jitc_vulkan_api_init_instance(VkInstance instance) {
    const char *symbol = nullptr;

    do {
        #define LOAD(name)                                                   \
            symbol = #name;                                                  \
            name = (PFN_##name) vkGetInstanceProcAddr(instance, #name);      \
            if (!name)                                                       \
                break;                                                       \
            symbol = nullptr

        LOAD(vkDestroyInstance);
        LOAD(vkEnumeratePhysicalDevices);
        LOAD(vkGetPhysicalDeviceProperties2);
        LOAD(vkGetPhysicalDeviceFeatures2);
        LOAD(vkGetPhysicalDeviceQueueFamilyProperties);
        LOAD(vkGetPhysicalDeviceMemoryProperties);
        LOAD(vkEnumerateDeviceExtensionProperties);
        LOAD(vkCreateDevice);
        LOAD(vkDestroyDevice);
        LOAD(vkGetDeviceQueue);
        LOAD(vkDeviceWaitIdle);
        LOAD(vkCreateBuffer);
        LOAD(vkDestroyBuffer);
        LOAD(vkGetBufferMemoryRequirements);
        LOAD(vkGetBufferDeviceAddress);
        LOAD(vkAllocateMemory);
        LOAD(vkFreeMemory);
        LOAD(vkBindBufferMemory);
        LOAD(vkMapMemory);
        LOAD(vkCreateCommandPool);
        LOAD(vkDestroyCommandPool);
        LOAD(vkAllocateCommandBuffers);
        LOAD(vkFreeCommandBuffers);
        LOAD(vkBeginCommandBuffer);
        LOAD(vkEndCommandBuffer);
        LOAD(vkQueueSubmit);
        LOAD(vkCreateSemaphore);
        LOAD(vkDestroySemaphore);
        LOAD(vkGetSemaphoreCounterValue);
        LOAD(vkWaitSemaphores);
        LOAD(vkSignalSemaphore);
        LOAD(vkCmdPipelineBarrier);
        LOAD(vkCmdCopyBuffer);
        LOAD(vkCmdFillBuffer);
        LOAD(vkCmdUpdateBuffer);
        LOAD(vkCmdBindPipeline);
        LOAD(vkCmdPushConstants);
        LOAD(vkCmdDispatch);
        LOAD(vkCreateShaderModule);
        LOAD(vkDestroyShaderModule);
        LOAD(vkCreatePipelineLayout);
        LOAD(vkDestroyPipelineLayout);
        LOAD(vkCreatePipelineCache);
        LOAD(vkDestroyPipelineCache);
        LOAD(vkGetPipelineCacheData);
        LOAD(vkCreateComputePipelines);
        LOAD(vkDestroyPipeline);

        #undef LOAD
    } while (false);

    if (symbol) {
        jitc_log(LogLevel::Warn,
                 "jit_vulkan_api_init(): could not find symbol \"%s\" -- "
                 "disabling Vulkan backend!", symbol);
        return false;
    }

    return true;
}

void jitc_vulkan_api_shutdown() {
    if (!jitc_vulkan_handle)
        return;

    #define Z(x) x = nullptr
    Z(vkGetInstanceProcAddr); Z(vkEnumerateInstanceVersion);
    Z(vkCreateInstance); Z(vkDestroyInstance); Z(vkEnumeratePhysicalDevices);
    Z(vkGetPhysicalDeviceProperties2); Z(vkGetPhysicalDeviceFeatures2);
    Z(vkGetPhysicalDeviceQueueFamilyProperties);
    Z(vkGetPhysicalDeviceMemoryProperties);
    Z(vkEnumerateDeviceExtensionProperties); Z(vkCreateDevice);
    Z(vkDestroyDevice); Z(vkGetDeviceQueue); Z(vkDeviceWaitIdle);
    Z(vkCreateBuffer); Z(vkDestroyBuffer); Z(vkGetBufferMemoryRequirements);
    Z(vkGetBufferDeviceAddress); Z(vkAllocateMemory); Z(vkFreeMemory);
    Z(vkBindBufferMemory); Z(vkMapMemory); Z(vkCreateCommandPool);
    Z(vkDestroyCommandPool); Z(vkAllocateCommandBuffers);
    Z(vkFreeCommandBuffers); Z(vkBeginCommandBuffer); Z(vkEndCommandBuffer);
    Z(vkQueueSubmit); Z(vkCreateSemaphore); Z(vkDestroySemaphore);
    Z(vkGetSemaphoreCounterValue); Z(vkWaitSemaphores); Z(vkSignalSemaphore);
    Z(vkCmdPipelineBarrier); Z(vkCmdCopyBuffer); Z(vkCmdFillBuffer);
    Z(vkCmdUpdateBuffer); Z(vkCmdBindPipeline); Z(vkCmdPushConstants);
    Z(vkCmdDispatch); Z(vkCreateShaderModule); Z(vkDestroyShaderModule);
    Z(vkCreatePipelineLayout); Z(vkDestroyPipelineLayout);
    Z(vkCreatePipelineCache); Z(vkDestroyPipelineCache);
    Z(vkGetPipelineCacheData); Z(vkCreateComputePipelines);
    Z(vkDestroyPipeline); Z(spvContextCreate); Z(spvContextDestroy);
    Z(spvTextToBinary); Z(spvBinaryDestroy); Z(spvDiagnosticDestroy);
    Z(spvValidateBinary);
    #undef Z

#if !defined(_WIN32)
    dlclose(jitc_vulkan_handle);
    if (jitc_spirv_tools_handle)
        dlclose(jitc_spirv_tools_handle);
#else
    FreeLibrary((HMODULE) jitc_vulkan_handle);
    if (jitc_spirv_tools_handle)
        FreeLibrary((HMODULE) jitc_spirv_tools_handle);
#endif

    jitc_vulkan_handle = nullptr;
    jitc_spirv_tools_handle = nullptr;
}

void vulkan_check_impl(VkResult errval, const char *file, const int line) {
    if (unlikely(errval != VK_SUCCESS))
        jitc_fail("vulkan_check(): API error %i in %s:%i.", (int) errval,
                  file, line);
}
