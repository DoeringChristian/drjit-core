/*
    src/vulkan.h -- Vulkan backend declarations

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"

#if defined(DRJIT_ENABLE_VULKAN)

struct ThreadState;
struct Kernel;

/// Initialize the Vulkan backend
extern bool jitc_vulkan_init();

/// Release all resources held by the Vulkan backend
extern void jitc_vulkan_shutdown();

/// Wait for all Vulkan work submitted on the current thread to complete.
/// Releases the Dr.Jit lock while waiting.
extern void jitc_vulkan_sync(ThreadState *ts);

/// Submit the current thread's pending command buffer without waiting for it.
extern void jitc_vulkan_flush(ThreadState *ts);

/// Wait until the command buffers submitted by all threads have completed.
extern void jitc_vulkan_sync_devices();

/// Compile the just-assembled kernel into a Kernel object. Units are cached by
/// content hash, and their SPIR-V and pipelines are also cached on disk.
extern bool jitc_vulkan_kernel_compile(ThreadState *ts, Kernel &kernel);

/// Free a previously compiled Vulkan kernel.
extern void jitc_vulkan_kernel_free(int device, Kernel &kernel);

/// Block until the timeline semaphore ``timeline`` of ``device`` reaches
/// ``value``. Called without holding the Dr.Jit lock.
extern void vulkan_wait(void *device, void *timeline, uint64_t value);

/// Invoke ``callback(payload)`` on the completion thread of ``device`` once
/// the work submitted to the device so far has finished
extern void jitc_vulkan_enqueue_callback(int device, void (*callback)(void *),
                                         void *payload);

// ---------------------------------------------------------------------

/// A buffer together with the memory backing it. VkDevice, VkBuffer, and
/// VkDeviceMemory handles are opaque to keep Vulkan types out of shared headers.
struct VulkanBuffer {
    void *device;
    void *buffer;
    void *memory;
};

/// Allocate a new buffer of ``size`` bytes on ``device`` in device-local or
/// host-visible (``shared``) memory. Returns an owned handle. The ``ptr_out``
/// argument returns the mapped CPU pointer (shared) or the buffer device
/// address.
///
/// The ``vulkan_*`` functions lack the ``jitc_`` prefix to indicate that they
/// are called *without* holding the Dr.Jit lock.
extern void *vulkan_buffer_new(int device, size_t size, bool shared,
                               void **ptr_out);

/// Allocate device-local memory for the VkImage ``image`` and bind it. Returns
/// the VkDeviceMemory and its size, or nullptr if the allocation failed.
extern void *jitc_vulkan_image_memory(int device, void *image,
                                      size_t *size_out);

/// Release a buffer handle allocated by ``vulkan_buffer_new``.
extern void vulkan_buffer_free(void *buffer);

/// Register a buffer for use with the find_buffer API shown below
extern void jitc_vulkan_register_buffer(void *ptr, void *buffer, size_t size);

/// Find the ``VulkanBuffer`` for a given pointer address and the byte offset
/// from its start. Returns nullptr if no registered allocation contains ``ptr``.
extern void *jitc_vulkan_find_buffer(void *ptr, size_t *offset_out);

/// Unregister a previously registered buffer
extern void *jitc_vulkan_unregister_buffer(void *ptr);

/// Return the device address of a pointer into a device or shared allocation
extern uint64_t jitc_vulkan_device_address(void *ptr);

#endif // defined(DRJIT_ENABLE_VULKAN)
