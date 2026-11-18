/*
    src/vulkan_eval.h -- SPIR-V code generation macros.

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

#define fmt(fmt, ...)                                                          \
    buffer.fmt_vulkan(count_args(__VA_ARGS__), fmt_strlen(fmt), fmt,           \
                      ##__VA_ARGS__)
#define put(...) buffer.put(__VA_ARGS__)
#define fmt_intrinsic(fmt, ...)                                                \
    do {                                                                       \
        size_t tmpoff = buffer.size();                                         \
        buffer.fmt_vulkan(count_args(__VA_ARGS__), fmt_strlen(fmt), fmt,       \
                          ##__VA_ARGS__);                                      \
        jitc_unit_capture_preamble(tmpoff);                                    \
    } while (0)

enum class VarType : uint32_t;
struct Variable;

// Helpers shared by the SPIR-V code generators (vulkan_eval.cpp,
// vulkan_scatter.cpp)

/// Declare the SPIR-V type of 'vt'
extern void jitc_vulkan_declare_type(VarType vt);

/// Declare the type of pointers to 'vt' in device memory
extern void jitc_vulkan_declare_ptr_type(VarType vt);

/// Declare the constant '%c_<type>_<value>' of type 'vt'
extern void jitc_vulkan_declare_const(VarType vt, uint32_t value);

/// Unsigned integer type with the same size as 'vt'
extern VarType jitc_vulkan_bin_type(VarType vt);

/// Compute a pointer '%<v>_<tag>p' to the element '%r<index>' of an array of
/// type 'vt' at address '%r<base><suffix>'. Booleans are stored as bytes. The
/// tag distinguishes the accesses of a variable (e.g. a gathered output).
extern void jitc_vulkan_render_element_ptr(const Variable *v, const char *tag,
                                           VarType vt, uint32_t base,
                                           const char *suffix, uint32_t index);

/// Load '%<v><suffix>' from the pointer '%<v>_<tag>p'
extern void jitc_vulkan_render_load(const Variable *v, const char *tag,
                                    const char *suffix);

/// Store ``value`` through the pointer '%<v>_<tag>p'
extern void jitc_vulkan_render_store(const Variable *v, const char *tag,
                                     const Variable *value);

/// Begin a structured conditional block that only executes for lanes with an
/// active ``mask``, and end it. Phi nodes following the block can merge values
/// from its end ('%<v>_d') and from the inactive lanes ('%<v>_f').
extern void jitc_vulkan_render_if(const Variable *v, const Variable *mask);
extern void jitc_vulkan_render_endif(const Variable *v);

/// Fail with an error message naming the unsupported variable
extern void jitc_vulkan_render_unhandled(const Variable *v);
