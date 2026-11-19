/*
    src/vulkan_array.h -- Functionality to create, read, and write
    variable arrays / Vulkan (SPIR-V) code generation component.

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

struct Variable;

/// Position in the code buffer of the first block of the generated function,
/// where 'Function' storage variables must be declared
extern size_t jitc_vulkan_locals_target;

extern void jitc_vulkan_render_array(Variable *v, Variable *pred);
extern void jitc_vulkan_render_array_init(Variable *v, Variable *pred,
                                          Variable *value);
extern void jitc_vulkan_render_array_read(Variable *v, Variable *source,
                                          Variable *mask, Variable *offset);
extern void jitc_vulkan_render_array_write(Variable *v, Variable *target,
                                           Variable *value, Variable *mask,
                                           Variable *offset);
extern void jitc_vulkan_render_array_select(Variable *v, Variable *mask,
                                            Variable *t, Variable *f);
extern void jitc_vulkan_render_array_memcpy_in(Variable *v);
extern void jitc_vulkan_render_array_memcpy_out(Variable *v);
