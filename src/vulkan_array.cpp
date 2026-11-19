/*
    src/vulkan_array.cpp -- Functionality to create, read, and write
    variable arrays / Vulkan (SPIR-V) code generation component.

    An array is a 'Function' storage variable '%arr_<reg>'. Like in the other
    backends, array variables that share storage share the register index.

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "vulkan_array.h"
#include "eval.h"
#include "internal.h"
#include "var.h"
#include "array.h"
#include "log.h"
#include "strbuf.h"
#include "vulkan_eval.h"

size_t jitc_vulkan_locals_target = 0;

/// Declare the array type of ``v`` and the pointer types to it and its
/// elements in 'Function' storage
static void jitc_vulkan_declare_array_type(const Variable *v) {
    uint32_t length = (uint32_t) v->array_length;
    jitc_vulkan_declare_type((VarType) v->type);
    jitc_vulkan_declare_const(VarType::UInt32, length);
    fmt_intrinsic("%array_$t_$u = OpTypeArray %$t %c_u32_$u", v, length, v,
                  length);
    fmt_intrinsic("%ptr_fn_array_$t_$u = OpTypePointer Function %array_$t_$u",
                  v, length, v, length);
    fmt_intrinsic("%ptr_fn_$t = OpTypePointer Function %$t", v, v);
}

/// Declare the storage '%arr_<reg>' of the array ``v`` in the first block
static void jitc_vulkan_render_array_alloc(const Variable *v) {
    jitc_vulkan_declare_array_type(v);
    size_t start = buffer.size();
    fmt("%arr_$u = OpVariable %ptr_fn_array_$t_$u Function\n", v->reg_index,
        v, (uint32_t) v->array_length);
    buffer.move_suffix(start, jitc_vulkan_locals_target);
}

/// Begin a structured loop over the array elements '%<v>_<tag>i'. The loop
/// consists of the blocks '%<v>_<tag>s' (entry), '%<v>_<tag>h' (header),
/// '%<v>_<tag>b' (body), '%<v>_<tag>k' (back edge), and '%<v>_<tag>d' (exit).
static void jitc_vulkan_render_array_loop_start(const Variable *v,
                                                const char *tag) {
    jitc_vulkan_declare_const(VarType::UInt32, 0);
    jitc_vulkan_declare_const(VarType::UInt32, (uint32_t) v->array_length);
    fmt("OpBranch %$v_$ss\n"
        "%$v_$ss = OpLabel\n"
        "OpBranch %$v_$sh\n"
        "%$v_$sh = OpLabel\n"
        "%$v_$si = OpPhi %u32 %c_u32_0 %$v_$ss %$v_$sn %$v_$sk\n"
        "%$v_$sc = OpULessThan %bool %$v_$si %c_u32_$u\n"
        "OpLoopMerge %$v_$sd %$v_$sk None\n"
        "OpBranchConditional %$v_$sc %$v_$sb %$v_$sd\n"
        "%$v_$sb = OpLabel\n",
        v, tag,
        v, tag,
        v, tag,
        v, tag,
        v, tag, v, tag, v, tag, v, tag,
        v, tag, v, tag, (uint32_t) v->array_length,
        v, tag, v, tag,
        v, tag, v, tag, v, tag,
        v, tag);
}

static void jitc_vulkan_render_array_loop_end(const Variable *v,
                                              const char *tag) {
    jitc_vulkan_declare_const(VarType::UInt32, 1);
    fmt("OpBranch %$v_$sk\n"
        "%$v_$sk = OpLabel\n"
        "%$v_$sn = OpIAdd %u32 %$v_$si %c_u32_1\n"
        "OpBranch %$v_$sh\n"
        "%$v_$sd = OpLabel\n",
        v, tag,
        v, tag,
        v, tag, v, tag,
        v, tag,
        v, tag);
}

/// Compute the pointer '%<v>_xp' to element '%<v>_xi' of the kernel
/// input/output array ``v``, which is stored as '[i * size + r0]'
static void jitc_vulkan_render_array_param_ptr(const Variable *v) {
    VarType vt = (VarType) v->type;
    if (vt == VarType::Bool)
        vt = VarType::UInt8;
    uint32_t tsize = type_size[(int) vt];
    jitc_vulkan_declare_ptr_type(vt);
    jitc_vulkan_declare_const(VarType::UInt64, tsize);
    fmt("%$v_xl = OpIMul %u32 %$v_xi %size\n"
        "%$v_xr = OpIAdd %u32 %$v_xl %r0\n"
        "%$v_xe = OpUConvert %u64 %$v_xr\n"
        "%$v_xo = OpIMul %u64 %$v_xe %c_u64_$u\n"
        "%$v_xg = OpIAdd %u64 %$v_pd %$v_xo\n"
        "%$v_xp = OpConvertUToPtr %ptr_$s %$v_xg\n"
        "%$v_xf = OpAccessChain %ptr_fn_$t %arr_$u %$v_xi\n",
        v, v,
        v, v,
        v, v,
        v, v, tsize,
        v, v, v,
        v, type_name_spirv[(int) vt], v,
        v, v, v->reg_index, v);
}

void jitc_vulkan_render_array(Variable *v, Variable *pred) {
    if (pred && pred->array_state != (uint32_t) ArrayState::Conflicted) {
        v->reg_index = pred->reg_index;
        return;
    }
    jitc_vulkan_render_array_alloc(v);
}

void jitc_vulkan_render_array_init(Variable *v, Variable *pred,
                                   Variable *value) {
    jitc_vulkan_declare_array_type(v);
    jitc_vulkan_render_array_loop_start(v, "n");
    fmt("%$v_ne = OpAccessChain %ptr_fn_$t %arr_$u %$v_ni\n"
        "OpStore %$v_ne %$v\n",
        v, v, pred->reg_index, v,
        v, value);
    jitc_vulkan_render_array_loop_end(v, "n");
    v->reg_index = pred->reg_index;
}

void jitc_vulkan_render_array_read(Variable *v, Variable *source,
                                   Variable *mask, Variable *offset) {
    jitc_vulkan_declare_array_type(source);
    bool is_masked = !mask->is_literal();

    if (is_masked)
        jitc_vulkan_render_if(v, mask);

    fmt("%$v_ae = OpAccessChain %ptr_fn_$t %arr_$u ", v, v, source->reg_index);
    if (offset) {
        fmt("%$v\n", offset);
    } else {
        jitc_vulkan_declare_const(VarType::UInt32, (uint32_t) v->literal);
        fmt("%c_u32_$u\n", (uint32_t) v->literal);
    }
    fmt("%$v$s = OpLoad %$t %$v_ae\n", v, is_masked ? "_al" : "", v, v);

    if (is_masked) {
        jitc_vulkan_render_endif(v);
        jitc_vulkan_declare_const((VarType) v->type, 0);
        fmt("%$v = OpPhi %$t %$v_al %$v_d %c_$t_0 %$v_f\n", v, v, v, v, v, v);
    }
}

void jitc_vulkan_render_array_write(Variable *v, Variable *target,
                                    Variable *value, Variable *mask,
                                    Variable *offset) {
    if (offset && offset->is_array())
        offset = nullptr;

    jitc_vulkan_declare_array_type(target);
    bool copy = target->array_state == (uint32_t) ArrayState::Conflicted,
         is_masked = !mask->is_literal();
    uint32_t target_buffer = target->reg_index;

    if (copy) {
        target_buffer = jitc_array_buffer(v)->reg_index;
        fmt("OpCopyMemory %arr_$u %arr_$u\n", target_buffer, target->reg_index);
    }

    if (is_masked)
        jitc_vulkan_render_if(v, mask);

    fmt("%$v_ae = OpAccessChain %ptr_fn_$t %arr_$u ", v, value, target_buffer);
    if (offset) {
        fmt("%$v\n", offset);
    } else {
        jitc_vulkan_declare_const(VarType::UInt32, (uint32_t) v->literal);
        fmt("%c_u32_$u\n", (uint32_t) v->literal);
    }
    fmt("OpStore %$v_ae %$v\n", v, value);

    if (is_masked)
        jitc_vulkan_render_endif(v);

    v->reg_index = target_buffer;
}

void jitc_vulkan_render_array_select(Variable *v, Variable *mask, Variable *t,
                                     Variable *f) {
    uint32_t reg_index = jitc_array_buffer(v)->reg_index;
    jitc_vulkan_declare_array_type(v);

    fmt("OpSelectionMerge %$v_se None\n"
        "OpBranchConditional %$v %$v_st %$v_sf\n"
        "%$v_st = OpLabel\n",
        v, mask, v, v, v);
    if (reg_index != t->reg_index)
        fmt("OpCopyMemory %arr_$u %arr_$u\n", reg_index, t->reg_index);
    fmt("OpBranch %$v_se\n"
        "%$v_sf = OpLabel\n", v, v);
    if (reg_index != f->reg_index)
        fmt("OpCopyMemory %arr_$u %arr_$u\n", reg_index, f->reg_index);
    fmt("OpBranch %$v_se\n"
        "%$v_se = OpLabel\n", v, v);

    v->reg_index = reg_index;
}

void jitc_vulkan_render_array_memcpy_in(Variable *v) {
    jitc_vulkan_render_array_alloc(v);
    jitc_vulkan_render_array_loop_start(v, "x");
    jitc_vulkan_render_array_param_ptr(v);
    jitc_vulkan_render_load(v, "x", "_xv");
    fmt("OpStore %$v_xf %$v_xv\n", v, v);
    jitc_vulkan_render_array_loop_end(v, "x");
}

void jitc_vulkan_render_array_memcpy_out(Variable *v) {
    jitc_vulkan_render_array_loop_start(v, "x");
    jitc_vulkan_render_array_param_ptr(v);
    fmt("%$v_xv = OpLoad %$t %$v_xf\n", v, v, v);
    if (jitc_is_bool(v)) {
        // Booleans are stored as bytes
        jitc_vulkan_declare_const(VarType::UInt8, 0);
        jitc_vulkan_declare_const(VarType::UInt8, 1);
        fmt("%$v_xw = OpSelect %u8 %$v_xv %c_u8_1 %c_u8_0\n"
            "OpStore %$v_xp %$v_xw Aligned 1\n",
            v, v, v, v);
    } else {
        fmt("OpStore %$v_xp %$v_xv Aligned $u\n", v, v,
            (uint32_t) type_size[v->type]);
    }
    jitc_vulkan_render_array_loop_end(v, "x");
}
