/*
    src/vulkan_tex.h -- hardware texture support for the Vulkan backend

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if defined(DRJIT_ENABLE_VULKAN)

#include <cstdint>
#include <cstddef>

struct VulkanDevice;

/// Create the texture heap of ``device`` holding ``size`` textures per
/// dimension (called by jitc_vulkan_init())
extern void jitc_vulkan_heap_init(VulkanDevice &device, uint32_t size);

/// Release the texture heap and samplers of ``device``
extern void jitc_vulkan_heap_shutdown(VulkanDevice &device);

// ---------------------------------------------------------------------
//  Public texture API
// ---------------------------------------------------------------------

extern void *jitc_vulkan_tex_create(size_t ndim, const size_t *shape,
                                    size_t n_channels, int format,
                                    int filter_mode, int wrap_mode,
                                    int writable, int srgb, size_t n_levels,
                                    int mip_filter, size_t max_aniso,
                                    int compression);
extern uintptr_t jitc_vulkan_tex_native_handle(const void *handle,
                                               size_t sub_index);
extern void jitc_vulkan_tex_get_shape(const void *handle, size_t *shape);
extern void jitc_vulkan_tex_get_indices(const void *handle,
                                        uint32_t *indices);
extern void jitc_vulkan_tex_memcpy_d2t(const void *src_ptr,
                                       void *dst_handle, size_t level);
extern void jitc_vulkan_tex_memcpy_t2d(const void *src_handle,
                                       void *dst_ptr);
extern void jitc_vulkan_tex_lookup(const void *handle, const uint32_t *pos,
                                   uint32_t active, uint32_t *out);
extern void jitc_vulkan_tex_destroy(void *handle);

#endif // defined(DRJIT_ENABLE_VULKAN)
