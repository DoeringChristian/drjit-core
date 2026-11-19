/*
    src/vulkan_tex.cpp -- hardware texture support for the Vulkan backend

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#if defined(DRJIT_ENABLE_VULKAN)

#include "vulkan_tex.h"
#include "vulkan.h"
#include "vulkan_api.h"
#include "vulkan_ts.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include "op.h"
#include "eval.h"
#include "tex.h"

#include <vector>

/**
 * Kernels cannot address textures through pointers. Each device therefore has
 * a texture heap: a descriptor set that stays bound to all kernels and that
 * holds arrays of 1D, 2D, and 3D textures (combined image samplers, bindings
 * 0-2). A texture is identified by its slot in the array of its dimension,
 * which plays the role of the CUDA texture object: kernels receive it as a
 * pointer literal and index the heap with it. Slot 0 is reserved.
 *
 * The descriptors of a slot are written when a texture is created, which the
 * 'update after bind' and 'update unused while pending' flags permit while
 * other kernels run. A released slot is reused once the work submitted so far
 * has finished.
 */
struct VulkanHeap {
    VkDevice device;

    /// Number of slots per dimension
    uint32_t size;

    /// Released slots, and the first slot that was never used
    std::vector<uint32_t> free;
    uint32_t next = 1;

    /// Samplers shared by all textures with the same settings. A driver may
    /// limit the number of samplers to a few thousand.
    std::vector<std::pair<uint32_t, VkSampler>> samplers;
};

/// Host-side state for one Vulkan hardware texture
struct VulkanTexture : TextureBase {
    int device;

    /// Store sRGB textures as RGBA8 (the only sRGB format with mandatory
    /// support for linear filtering)?
    bool srgb_rgba;

    /// Per sub-texture: the VkImage, its VkImageView and VkDeviceMemory, and
    /// the variable holding its heap slot
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkDeviceMemory> memory;
    std::vector<uint32_t> indices;

    VulkanTexture(int device, size_t type_size, size_t n_channels,
                  bool srgb_rgba)
        : TextureBase(JitBackend::Vulkan, type_size, n_channels, false),
          device(device), srgb_rgba(srgb_rgba), images(n_textures),
          views(n_textures), memory(n_textures), indices(n_textures) { }

    /// Channels of the image of sub-texture ``index``
    size_t channels_image(size_t index) const {
        return srgb_rgba ? 4 : channels_storage(index);
    }

    /// Returns the ID of a pointer literal that holds the slot of sub-texture
    /// ``i``, analogous to CUDATexture::get_jit_pointer()
    uint32_t get_jit_pointer(size_t i) const {
        return jitc_var_pointer(JitBackend::Vulkan, jitc_var(indices[i])->data,
                                indices[i], false);
    }
};

/// The resources of a sub-texture, released once the device is done with them
struct VulkanTexRelease {
    VulkanHeap *heap;
    VkImage image;
    VkImageView view;
    VkDeviceMemory memory;
    uint32_t slot;
};

// ============================================================================
//  Texture heap
// ============================================================================

void jitc_vulkan_heap_init(VulkanDevice &device, uint32_t size) {
    VkDevice dev = (VkDevice) device.device;

    VkDescriptorSetLayoutBinding bindings[3] { };
    VkDescriptorBindingFlags flags[3];
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = size;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        flags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                   VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                   VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
    flags_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flags_info.bindingCount = 3;
    flags_info.pBindingFlags = flags;

    VkDescriptorSetLayoutCreateInfo layout_info {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.pNext = &flags_info;
    layout_info.flags =
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layout_info.bindingCount = 3;
    layout_info.pBindings = bindings;
    vulkan_check(vkCreateDescriptorSetLayout(
        dev, &layout_info, nullptr,
        (VkDescriptorSetLayout *) &device.heap_layout));

    VkDescriptorPoolSize pool_size { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                     3 * size };
    VkDescriptorPoolCreateInfo pool_info {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    vulkan_check(vkCreateDescriptorPool(dev, &pool_info, nullptr,
                                        (VkDescriptorPool *) &device.heap_pool));

    VkDescriptorSetAllocateInfo set_info {};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = (VkDescriptorPool) device.heap_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = (VkDescriptorSetLayout *) &device.heap_layout;
    vulkan_check(vkAllocateDescriptorSets(dev, &set_info,
                                          (VkDescriptorSet *) &device.heap));

    VulkanHeap *heap = new VulkanHeap();
    heap->device = dev;
    heap->size = size;
    device.heap_state = heap;
}

void jitc_vulkan_heap_shutdown(VulkanDevice &device) {
    VkDevice dev = (VkDevice) device.device;
    VulkanHeap *heap = (VulkanHeap *) device.heap_state;
    for (auto &[key, sampler] : heap->samplers)
        vkDestroySampler(dev, sampler, nullptr);
    delete heap;

    // Destroying the pool frees the descriptor set
    vkDestroyDescriptorPool(dev, (VkDescriptorPool) device.heap_pool, nullptr);
    vkDestroyDescriptorSetLayout(
        dev, (VkDescriptorSetLayout) device.heap_layout, nullptr);
}

static uint32_t vulkan_heap_alloc(VulkanHeap *heap) {
    if (!heap->free.empty()) {
        uint32_t slot = heap->free.back();
        heap->free.pop_back();
        return slot;
    }
    if (heap->next == heap->size)
        jitc_raise("jit_tex_create(): the texture heap of the Vulkan device is "
                   "full (%u textures per dimension)!", heap->size - 1);
    return heap->next++;
}

/// Release the resources of a sub-texture. Runs on the completion thread once
/// the device has executed the work submitted before the release.
static void vulkan_tex_release(void *payload) {
    VulkanTexRelease *r = (VulkanTexRelease *) payload;
    VkDevice dev = r->heap->device;
    vkDestroyImageView(dev, r->view, nullptr);
    vkDestroyImage(dev, r->image, nullptr);
    vkFreeMemory(dev, r->memory, nullptr);

    /* The heap is protected by the Dr.Jit lock */ {
        lock_guard guard(state.lock);
        r->heap->free.push_back(r->slot);
    }

    delete r;
}

// ============================================================================
//  Creation / destruction
// ============================================================================

static VkFormat vulkan_tex_format(int format, size_t channels, bool srgb) {
    // Validated by jitc_vulkan_tex_create()
    switch ((VarType) format) {
        case VarType::UInt8:
            if (srgb)
                return VK_FORMAT_R8G8B8A8_SRGB;
            switch (channels) {
                case 1: return VK_FORMAT_R8_UNORM;
                case 2: return VK_FORMAT_R8G8_UNORM;
                default: return VK_FORMAT_R8G8B8A8_UNORM;
            }
        case VarType::Float16:
            switch (channels) {
                case 1: return VK_FORMAT_R16_SFLOAT;
                case 2: return VK_FORMAT_R16G16_SFLOAT;
                default: return VK_FORMAT_R16G16B16A16_SFLOAT;
            }
        default:
            switch (channels) {
                case 1: return VK_FORMAT_R32_SFLOAT;
                case 2: return VK_FORMAT_R32G32_SFLOAT;
                default: return VK_FORMAT_R32G32B32A32_SFLOAT;
            }
    }
}

/// Return the shared sampler with the given settings, creating it on first use
static VkSampler vulkan_tex_sampler(VulkanHeap *heap, int filter_mode,
                                    int wrap_mode, int mip_filter) {
    VkFilter filter;
    switch (filter_mode) {
        case 0: filter = VK_FILTER_NEAREST; break;
        case 1: filter = VK_FILTER_LINEAR; break;
        default: jitc_raise("jit_tex_create(): invalid filter mode!");
    }

    VkSamplerAddressMode wrap;
    switch (wrap_mode) {
        case 0: wrap = VK_SAMPLER_ADDRESS_MODE_REPEAT; break;
        case 1: wrap = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; break;
        case 2: wrap = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; break;
        default: jitc_raise("jit_tex_create(): invalid wrap mode!");
    }

    uint32_t key = (uint32_t) filter_mode | ((uint32_t) wrap_mode << 1) |
                   ((uint32_t) (mip_filter != 0) << 3);
    for (auto &[k, sampler] : heap->samplers)
        if (k == key)
            return sampler;

    VkSamplerCreateInfo sampler_info {};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = filter;
    sampler_info.minFilter = filter;
    sampler_info.mipmapMode = mip_filter == 0
                                  ? VK_SAMPLER_MIPMAP_MODE_NEAREST
                                  : VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = wrap;
    sampler_info.addressModeV = wrap;
    sampler_info.addressModeW = wrap;
    sampler_info.maxLod = VK_LOD_CLAMP_NONE;

    VkSampler sampler = VK_NULL_HANDLE;
    vulkan_check(vkCreateSampler(heap->device, &sampler_info, nullptr,
                                 &sampler));
    heap->samplers.emplace_back(key, sampler);
    return sampler;
}

/// Install a callback on sub-texture ``i`` that releases its resources and
/// (once the last sub-texture is freed) the ``VulkanTexture`` itself
static void vulkan_tex_install_release(VulkanTexture *tex, size_t i,
                                       VulkanTexRelease *release) {
    struct Payload {
        VulkanTexture *texture;
        VulkanTexRelease *release;
    };

    jitc_var_set_callback(
        tex->indices[i],
        [](uint32_t, int free, void *cb) {
            if (!free)
                return;
            Payload *p = (Payload *) cb;
            VulkanTexture *t = p->texture;

            // Only release the resources if the device is still alive. Kernels
            // that may still use them are submitted first.
            if (state.backends & (1u << (uint32_t) JitBackend::Vulkan)) {
                jitc_vulkan_flush(thread_state(JitBackend::Vulkan));
                jitc_vulkan_enqueue_callback(t->device, vulkan_tex_release,
                                             p->release);
            } else {
                delete p->release;
            }

            if (--t->n_referenced_textures == 0)
                delete t;
            delete p;
        },
        new Payload{ tex, release }, false);
}

void *jitc_vulkan_tex_create(size_t ndim, const size_t *shape,
                             size_t n_channels, int format, int filter_mode,
                             int wrap_mode, int writable, int srgb,
                             size_t n_levels, int mip_filter,
                             size_t /* max_aniso */, int compression) {
    if (ndim < 1 || ndim > 3)
        jitc_raise("jit_tex_create(): invalid texture dimension!");
    else if (n_channels == 0)
        jitc_raise("jit_tex_create(): must have at least 1 channel!");
    for (size_t i = 0; i < ndim; ++i)
        if (shape[i] == 0)
            jitc_raise("jit_tex_create(): texture dimensions must be nonzero!");
    if (n_levels > 1)
        jitc_raise("jit_tex_create(): MIP-mapped textures are not yet "
                   "supported by the Vulkan backend.");
    if (writable)
        jitc_raise("jit_tex_create(): writable textures are not yet supported "
                   "by the Vulkan backend.");
    if (compression)
        jitc_raise("jit_tex_create(): block-compressed textures are not yet "
                   "supported by the Vulkan backend.");

    size_t type_size;
    switch ((VarType) format) {
        case VarType::Float32: type_size = sizeof(float); break;
        case VarType::Float16: type_size = sizeof(uint16_t); break;
        case VarType::UInt8:   type_size = sizeof(uint8_t); break;
        default: jitc_raise("jit_tex_create(): invalid data type!");
    }

    ThreadState *ts = thread_state(JitBackend::Vulkan);
    VulkanThreadState *vts = (VulkanThreadState *) ts->actual_state();
    const VulkanDevice &dev = state.vulkan_devices[ts->device];
    VkDevice device = (VkDevice) dev.device;
    VulkanHeap *heap = (VulkanHeap *) dev.heap_state;
    VkSampler sampler =
        vulkan_tex_sampler(heap, filter_mode, wrap_mode, mip_filter);

    bool srgb_rgba = srgb && (VarType) format == VarType::UInt8;
    VulkanTexture *tex =
        new VulkanTexture(ts->device, type_size, n_channels, srgb_rgba);
    tex->ndim = ndim;
    for (size_t i = 0; i < ndim; ++i)
        tex->shape[i] = shape[i];

    static const VkImageType image_types[3] = {
        VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D
    };
    static const VkImageViewType view_types[3] = {
        VK_IMAGE_VIEW_TYPE_1D, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_3D
    };

    size_t total_bytes = 0;
    for (size_t i = 0; i < tex->n_textures; ++i) {
        VkFormat vk_format =
            vulkan_tex_format(format, tex->channels_image(i), srgb_rgba);

        VkImageCreateInfo image_info {};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = image_types[ndim - 1];
        image_info.format = vk_format;
        image_info.extent = { (uint32_t) shape[0],
                              ndim >= 2 ? (uint32_t) shape[1] : 1u,
                              ndim == 3 ? (uint32_t) shape[2] : 1u };
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage image = VK_NULL_HANDLE;
        size_t bytes = 0;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS ||
            !(memory = (VkDeviceMemory) jitc_vulkan_image_memory(
                  ts->device, image, &bytes)))
            jitc_raise("jit_tex_create(): texture allocation failed!");
        total_bytes += bytes;

        VkImageViewCreateInfo view_info {};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = image;
        view_info.viewType = view_types[ndim - 1];
        view_info.format = vk_format;
        view_info.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkImageView view = VK_NULL_HANDLE;
        vulkan_check(vkCreateImageView(device, &view_info, nullptr, &view));

        // Images stay in the general layout, which permits sampling and copies
        VkImageMemoryBarrier barrier {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = view_info.subresourceRange;
        vkCmdPipelineBarrier((VkCommandBuffer) vts->ensure_cmdbuf(),
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &barrier);

        uint32_t slot = vulkan_heap_alloc(heap);
        VkDescriptorImageInfo descr { sampler, view, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet write {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = (VkDescriptorSet) dev.heap;
        write.dstBinding = (uint32_t) ndim - 1;
        write.dstArrayElement = slot;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &descr;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

        tex->images[i] = image;
        tex->views[i] = view;
        tex->memory[i] = memory;
        tex->indices[i] = jitc_var_mem_map(JitBackend::Vulkan, VarType::UInt64,
                                           (void *) (uintptr_t) slot, 1, 0);
        vulkan_tex_install_release(
            tex, i, new VulkanTexRelease{ heap, image, view, memory, slot });
    }

    tex->device_bytes = total_bytes;
    jitc_tex_track_usage(JitBackend::Vulkan, (int64_t) total_bytes, 1);

    jitc_log(LogLevel::Debug, "jitc_vulkan_tex_create(): " DRJIT_PTR,
             (uintptr_t) tex);
    return (void *) tex;
}

uintptr_t jitc_vulkan_tex_native_handle(const void *handle,
                                        size_t sub_index) {
    const VulkanTexture &tex = *((const VulkanTexture *) handle);
    if (sub_index >= tex.n_textures)
        jitc_raise("jit_tex_native_handle(): sub-texture index out of range.");
    return (uintptr_t) tex.images[sub_index];
}

void jitc_vulkan_tex_get_shape(const void *handle, size_t *shape) {
    const VulkanTexture &tex = *((const VulkanTexture *) handle);
    for (size_t i = 0; i < tex.ndim; ++i)
        shape[i] = tex.shape[i];
    shape[tex.ndim] = tex.n_channels;
}

void jitc_vulkan_tex_get_indices(const void *handle, uint32_t *indices) {
    if (!handle)
        return;
    const VulkanTexture &tex = *((const VulkanTexture *) handle);
    for (size_t i = 0; i < tex.n_textures; ++i)
        indices[i] = tex.indices[i];
}

void jitc_vulkan_tex_destroy(void *handle) {
    if (!handle)
        return;
    jitc_log(LogLevel::Debug, "jitc_vulkan_tex_destroy(" DRJIT_PTR ")",
             (uintptr_t) handle);

    VulkanTexture *tex = (VulkanTexture *) handle;

    // Fetch the count in case the struct is deleted during the traversal
    const size_t n_textures = tex->n_textures;
    for (size_t i = 0; i < n_textures; ++i)
        jitc_var_dec_ref(tex->indices[i]);
}

// ============================================================================
//  Device <-> texture memory transfer
// ============================================================================

/// Launch a kernel to (de)interleave texture memory, see the Metal version
/// ``metal_channel_pack()``
static void vulkan_channel_pack(VulkanThreadState *vts, VulkanKernel kernel,
                                const void *src, void *dst,
                                uint32_t n_channels, uint32_t ci,
                                uint32_t tex_base, uint32_t c_valid,
                                uint32_t size) {
    struct {
        uint64_t src, dst;
        uint32_t n_channels, ci, tex_base, c_valid, size;
    } params { (uint64_t) (uintptr_t) src, (uint64_t) (uintptr_t) dst,
               n_channels, ci, tex_base, c_valid, size };
    jitc_vulkan_launch_utility(vts, kernel, &params, sizeof(params), size);
}

/// Copy between the image of sub-texture ``i`` and the device memory ``ptr``
static void vulkan_tex_copy(VulkanThreadState *vts, const VulkanTexture &tex,
                            size_t i, void *ptr, bool to_texture) {
    size_t offset = 0;
    VulkanBuffer *buf = (VulkanBuffer *) jitc_vulkan_find_buffer(ptr, &offset);
    if (!buf)
        jitc_raise("jit_tex_memcpy(): " DRJIT_PTR " is not a Vulkan "
                   "allocation!", (uintptr_t) ptr);

    VkBufferImageCopy region {};
    region.bufferOffset = offset;
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { (uint32_t) tex.shape[0],
                           tex.ndim >= 2 ? (uint32_t) tex.shape[1] : 1u,
                           tex.ndim == 3 ? (uint32_t) tex.shape[2] : 1u };

    VkCommandBuffer cb = (VkCommandBuffer) vts->ensure_cmdbuf();
    if (to_texture)
        vkCmdCopyBufferToImage(cb, (VkBuffer) buf->buffer, tex.images[i],
                               VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    else
        vkCmdCopyImageToBuffer(cb, tex.images[i], VK_IMAGE_LAYOUT_GENERAL,
                               (VkBuffer) buf->buffer, 1, &region);
}

/// Pick the (de)interleave kernel for elements of ``type_size`` bytes
static VulkanKernel vulkan_tex_pack_kernel(bool deinterleave,
                                           size_t type_size) {
    uint32_t base = deinterleave ? (uint32_t) VulkanKernel::DeinterleaveU8
                                 : (uint32_t) VulkanKernel::InterleaveU8;
    return (VulkanKernel) (base + (type_size == 1 ? 0 : type_size == 2 ? 1 : 2));
}

void jitc_vulkan_tex_memcpy_d2t(const void *src_ptr, void *dst_handle,
                                size_t level) {
    VulkanTexture &tex = *((VulkanTexture *) dst_handle);
    if (level >= tex.n_levels)
        jitc_raise("jit_tex_memcpy_d2t(): MIP level %zu out of range!", level);

    ThreadState *ts = thread_state(JitBackend::Vulkan);
    VulkanThreadState *vts = (VulkanThreadState *) ts->actual_state();

    size_t n_texels = 1, type_size = tex.type_size;
    for (size_t i = 0; i < tex.ndim; ++i)
        n_texels *= tex.shape[i];

    // Interleaved channels are split into the packed, zero-padded layouts of
    // the sub-textures, unless a single image holds them as they are
    bool needs_staging = tex.n_textures > 1 ||
                         tex.channels_image(0) != tex.n_channels;
    VulkanKernel kernel = vulkan_tex_pack_kernel(true, type_size);

    for (size_t i = 0; i < tex.n_textures; ++i) {
        if (!needs_staging) {
            vulkan_tex_copy(vts, tex, i, (void *) src_ptr, true);
            continue;
        }
        size_t ci = tex.channels_image(i);
        void *packed = jitc_malloc(JitBackend::Vulkan,
                                   n_texels * ci * type_size);
        vulkan_channel_pack(vts, kernel, src_ptr, packed,
                            (uint32_t) tex.n_channels, (uint32_t) ci,
                            (uint32_t) (i * 4), (uint32_t) tex.channels(i),
                            (uint32_t) (n_texels * ci));
        vulkan_tex_copy(vts, tex, i, packed, true);
        jitc_free(packed);
    }
}

void jitc_vulkan_tex_memcpy_t2d(const void *src_handle, void *dst_ptr) {
    const VulkanTexture &tex = *((const VulkanTexture *) src_handle);
    ThreadState *ts = thread_state(JitBackend::Vulkan);
    VulkanThreadState *vts = (VulkanThreadState *) ts->actual_state();

    size_t n_texels = 1, type_size = tex.type_size;
    for (size_t i = 0; i < tex.ndim; ++i)
        n_texels *= tex.shape[i];

    bool needs_staging = tex.n_textures > 1 ||
                         tex.channels_image(0) != tex.n_channels;
    VulkanKernel kernel = vulkan_tex_pack_kernel(false, type_size);

    for (size_t i = 0; i < tex.n_textures; ++i) {
        if (!needs_staging) {
            vulkan_tex_copy(vts, tex, i, dst_ptr, false);
            continue;
        }
        size_t ci = tex.channels_image(i), c = tex.channels(i);
        void *packed = jitc_malloc(JitBackend::Vulkan,
                                   n_texels * ci * type_size);
        vulkan_tex_copy(vts, tex, i, packed, false);
        vulkan_channel_pack(vts, kernel, packed, dst_ptr,
                            (uint32_t) tex.n_channels, (uint32_t) ci,
                            (uint32_t) (i * 4), (uint32_t) c,
                            (uint32_t) (n_texels * c));
        jitc_free(packed);
    }
}

// ============================================================================
//  Lookup graph builders
// ============================================================================

/// Validate the coordinates of a lookup, see ``jitc_cuda_tex_check()``
static Variable jitc_vulkan_tex_check(size_t ndim, const uint32_t *pos) {
    uint32_t size = 0;
    bool dirty = false, symbolic = false;

    for (size_t i = 0; i < ndim; ++i) {
        const Variable *v = jitc_var(pos[i]);
        if ((VarType) v->type != VarType::Float32)
            jitc_raise("jit_tex_lookup(): type mismatch for arg. %zu (got %s, "
                       "expected %s)", i, type_name[v->type],
                       type_name[(int) VarType::Float32]);
        size = std::max(size, v->size);
        dirty |= v->is_dirty();
        symbolic |= (bool) v->symbolic;
    }

    for (size_t i = 0; i < ndim; ++i) {
        const Variable *v = jitc_var(pos[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise("jit_tex_lookup(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if (dirty) {
        jitc_eval(thread_state(JitBackend::Vulkan));
        for (size_t i = 0; i < ndim; ++i) {
            if (jitc_var(pos[i])->is_dirty())
                jitc_raise_dirty_error(pos[i]);
        }
    }

    Variable v;
    v.size = size;
    v.backend = (uint32_t) JitBackend::Vulkan;
    v.symbolic = symbolic;
    v.type = (uint32_t) VarType::Float32;
    return v;
}

/// Emit a ``TexLookup`` node per sub-texture with the same layout as the CUDA
/// backend: dep[0] is the slot (zero in masked lanes, with the literal set),
/// followed by the coordinates
void jitc_vulkan_tex_lookup(const void *handle, const uint32_t *pos,
                            uint32_t active, uint32_t *out) {
    const VulkanTexture &tex = *((const VulkanTexture *) handle);
    size_t ndim = tex.ndim;
    Variable v = jitc_vulkan_tex_check(ndim, pos);

    const Variable *active_v = jitc_var(active);
    bool masked = !(active_v->is_literal() && active_v->literal == 1);

    for (size_t ti = 0; ti < tex.n_textures; ++ti) {
        v.kind = (uint32_t) VarKind::TexLookup;
        memset(v.dep, 0, sizeof(v.dep));
        Ref pointer = steal(tex.get_jit_pointer(ti));
        if (masked) {
            v.literal = 1;
            uint64_t zero_i = 0;
            Ref zero = steal(jitc_var_literal(JitBackend::Vulkan,
                                              VarType::Pointer, &zero_i, 1, 0));
            v.dep[0] = jitc_var_select(active, pointer, zero);
        } else {
            v.literal = 0;
            v.dep[0] = pointer.release();
        }
        for (size_t j = 0; j < ndim; ++j) {
            v.dep[j + 1] = pos[j];
            jitc_var_inc_ref(pos[j]);
        }
        Ref tex_load = steal(jitc_var_new(v));

        v.kind = (uint32_t) VarKind::Extract;
        memset(v.dep, 0, sizeof(v.dep));
        for (size_t ch = 0; ch < tex.channels(ti); ++ch) {
            v.literal = (uint64_t) ch;
            v.dep[0] = tex_load;
            jitc_var_inc_ref(tex_load);
            *out++ = jitc_var_new(v);
        }
    }
}

#endif // defined(DRJIT_ENABLE_VULKAN)
