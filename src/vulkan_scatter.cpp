/*
    src/vulkan_scatter.cpp -- Scatter / scatter-reduce SPIR-V codegen.

    The atomic operations below use device scope ('%c_u32_1') and relaxed
    memory semantics ('%c_u32_0'), like the 'red' and 'atom' instructions of
    the CUDA backend.

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "vulkan_scatter.h"
#include "eval.h"
#include "internal.h"
#include "var.h"
#include "op.h"
#include "log.h"
#include "util.h"
#include "strbuf.h"
#include "vulkan_eval.h"

/// Declare the scope and memory semantics operands of atomic operations
static void jitc_vulkan_declare_atomic_operands() {
    jitc_vulkan_declare_const(VarType::UInt32, 0);
    jitc_vulkan_declare_const(VarType::UInt32, 1);
}

/// Vulkan provides atomics for 32- and 64-bit values
static void jitc_vulkan_check_atomic(const char *name, const Variable *value) {
    uint32_t tsize = type_size[value->type];
    if (tsize != 4 && tsize != 8)
        jitc_raise("%s(): atomic operations on variables of type %s are not "
                   "supported by the Vulkan backend.", name,
                   type_name[value->type]);
}

/// Opcode for an unsigned view of a value of the same size
static const char *jitc_vulkan_bin_cast(const Variable *v) {
    return jitc_is_uint(v) ? "OpCopyObject" : "OpBitcast";
}

/// Vulkan devices generally lack floating point min/max atomics. Reduce them
/// to integer ones like the CUDA backend (see jitc_cuda_render_atomic_minmax()),
/// exploiting that IEEE-754 bit patterns order like signed integers when
/// non-negative, and like unsigned integers in reverse otherwise. NaN values
/// are ignored, like LLVM's 'atomicrmw fmin/fmax' and the warp-local
/// pre-reduction of the CUDA backend.
static void jitc_vulkan_render_atomic_minmax(const Variable *v,
                                             const Variable *value,
                                             ReduceOp op) {
    bool is64 = type_size[value->type] == 8,
         is_min = op == ReduceOp::Min;
    VarType st = is64 ? VarType::Int64 : VarType::Int32,
            ut = is64 ? VarType::UInt64 : VarType::UInt32;
    const char *s = type_name_spirv[(int) st],
               *u = type_name_spirv[(int) ut];

    jitc_vulkan_declare_ptr_type(st);
    jitc_vulkan_declare_ptr_type(ut);
    jitc_vulkan_declare_const(st, 0);

    fmt("%$v_i = OpIsNan %bool %$v\n"
        "OpSelectionMerge %$v_k None\n"
        "OpBranchConditional %$v_i %$v_k %$v_j\n"
        "%$v_j = OpLabel\n",
        v, value,
        v,
        v, v, v,
        v);

    fmt("%$v_s = OpBitcast %$s %$v\n"
        "%$v_n = OpSLessThan %bool %$v_s %c_$s_0\n"
        "%$v_sp = OpConvertUToPtr %ptr_$s %$v_ga\n"
        "%$v_up = OpConvertUToPtr %ptr_$s %$v_ga\n"
        "OpSelectionMerge %$v_m None\n"
        "OpBranchConditional %$v_n %$v_q %$v_p\n"
        "%$v_p = OpLabel\n"
        "%$v_a = $s %$s %$v_sp %c_u32_1 %c_u32_0 %$v_s\n"
        "OpBranch %$v_m\n"
        "%$v_q = OpLabel\n"
        "%$v_u = OpBitcast %$s %$v\n"
        "%$v_b = $s %$s %$v_up %c_u32_1 %c_u32_0 %$v_u\n"
        "OpBranch %$v_m\n"
        "%$v_m = OpLabel\n"
        "OpBranch %$v_k\n"
        "%$v_k = OpLabel\n",
        v, s, value,
        v, v, s,
        v, s, v,
        v, u, v,
        v,
        v, v, v,
        v,
        v, is_min ? "OpAtomicSMin" : "OpAtomicSMax", s, v, v,
        v,
        v,
        v, u, value,
        v, is_min ? "OpAtomicUMax" : "OpAtomicUMin", u, v, v,
        v,
        v,
        v,
        v);
}

void jitc_vulkan_render_scatter(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]),
             *value = jitc_var(v->dep[1]),
             *index = jitc_var(v->dep[2]),
             *mask  = jitc_var(v->dep[3]);

    // All reduction modes map to atomics (no subgroup aggregation yet)
    ReduceOp op = (ReduceOp) (uint32_t) v->literal;
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    jitc_vulkan_render_element_ptr(v, "g", (VarType) value->type,
                                   ptr->reg_index, "", index->reg_index);

    if (!is_unmasked)
        jitc_vulkan_render_if(v, mask);

    if (op == ReduceOp::Identity) {
        jitc_vulkan_render_store(v, "g", value);
    } else {
        jitc_vulkan_check_atomic("jitc_vulkan_render_scatter", value);
        jitc_vulkan_declare_atomic_operands();

        if (jitc_is_float(value) && (op == ReduceOp::Min || op == ReduceOp::Max)) {
            jitc_vulkan_render_atomic_minmax(v, value, op);
        } else {
            bool is_sint = jitc_is_sint(value);
            const char *name;
            switch (op) {
                case ReduceOp::Add:
                    name = jitc_is_float(value) ? "OpAtomicFAddEXT" : "OpAtomicIAdd";
                    break;
                case ReduceOp::Min: name = is_sint ? "OpAtomicSMin" : "OpAtomicUMin"; break;
                case ReduceOp::Max: name = is_sint ? "OpAtomicSMax" : "OpAtomicUMax"; break;
                case ReduceOp::And: name = "OpAtomicAnd"; break;
                case ReduceOp::Or:  name = "OpAtomicOr"; break;
                default:
                    jitc_fail("jitc_vulkan_render_scatter(): unsupported "
                              "reduction (%s).", red_name[(int) op]);
            }
            fmt("%$v_r = $s %$t %$v_gp %c_u32_1 %c_u32_0 %$v\n", v, name,
                value, v, value);
        }
    }

    if (!is_unmasked)
        jitc_vulkan_render_endif(v);
}

void jitc_vulkan_render_scatter_inc(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]),
             *index = jitc_var(v->dep[1]),
             *mask  = jitc_var(v->dep[2]);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    jitc_vulkan_render_element_ptr(v, "g", VarType::UInt32, ptr->reg_index,
                                   "", index->reg_index);
    jitc_vulkan_declare_atomic_operands();

    if (is_unmasked) {
        fmt("%$v = OpAtomicIAdd %u32 %$v_gp %c_u32_1 %c_u32_0 %c_u32_1\n",
            v, v);
    } else {
        jitc_vulkan_render_if(v, mask);
        fmt("%$v_l = OpAtomicIAdd %u32 %$v_gp %c_u32_1 %c_u32_0 %c_u32_1\n",
            v, v);
        jitc_vulkan_render_endif(v);
        fmt("%$v = OpPhi %u32 %$v_l %$v_d %c_u32_0 %$v_f\n", v, v, v, v);
    }

    v->consumed = 1;
}

void jitc_vulkan_render_scatter_exch(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]),
             *value = jitc_var(v->dep[1]),
             *index = jitc_var(v->dep[2]),
             *mask  = jitc_var(v->dep[3]);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    const char *cast = jitc_vulkan_bin_cast(value);

    jitc_vulkan_check_atomic("jitc_vulkan_render_scatter_exch", value);
    jitc_vulkan_render_element_ptr(v, "g", jitc_vulkan_bin_type((VarType) value->type),
                                   ptr->reg_index, "", index->reg_index);
    jitc_vulkan_declare_atomic_operands();

    if (!is_unmasked)
        jitc_vulkan_render_if(v, mask);

    fmt("%$v_x = $s %$b %$v\n"
        "%$v_r = OpAtomicExchange %$b %$v_gp %c_u32_1 %c_u32_0 %$v_x\n"
        "%$v$s = $s %$t %$v_r\n",
        v, cast, value, value,
        v, value, v, v,
        v, is_unmasked ? "" : "_l", cast, v, v);

    if (!is_unmasked) {
        jitc_vulkan_render_endif(v);
        jitc_vulkan_declare_const((VarType) v->type, 0);
        fmt("%$v = OpPhi %$t %$v_l %$v_d %c_$t_0 %$v_f\n", v, v, v, v, v, v);
    }

    v->consumed = 1;
}

void jitc_vulkan_render_scatter_cas(Variable *v) {
    Variable *ptr     = jitc_var(v->dep[0]),
             *compare = jitc_var(v->dep[1]),
             *value   = jitc_var(v->dep[2]),
             *index   = jitc_var(v->dep[3]);

    ScatterCASDData *cas_data = (ScatterCASDData *) v->data;
    Variable *mask = jitc_var(cas_data->mask);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    const char *cast = jitc_vulkan_bin_cast(value);

    jitc_vulkan_check_atomic("jitc_vulkan_render_scatter_cas", value);
    jitc_vulkan_render_element_ptr(v, "g", jitc_vulkan_bin_type((VarType) value->type),
                                   ptr->reg_index, "", index->reg_index);
    jitc_vulkan_declare_atomic_operands();

    if (!is_unmasked)
        jitc_vulkan_render_if(v, mask);

    // Outputs: the old value ('%<v>_out_0') and whether the swap took place
    // ('%<v>_out_1'), zero/false for masked lanes
    const char *s0 = is_unmasked ? "_out_0" : "_o",
               *s1 = is_unmasked ? "_out_1" : "_s";
    fmt("%$v_c = $s %$b %$v\n"
        "%$v_x = $s %$b %$v\n"
        "%$v_r = OpAtomicCompareExchange %$b %$v_gp %c_u32_1 %c_u32_0 %c_u32_0 %$v_x %$v_c\n"
        "%$v$s = OpIEqual %bool %$v_r %$v_c\n"
        "%$v$s = $s %$t %$v_r\n",
        v, cast, compare, compare,
        v, cast, value, value,
        v, value, v, v, v,
        v, s1, v, v,
        v, s0, cast, value, v);

    if (!is_unmasked) {
        jitc_vulkan_render_endif(v);
        jitc_vulkan_declare_const((VarType) value->type, 0);
        jitc_vulkan_declare_const(VarType::Bool, 0);
        fmt("%$v_out_0 = OpPhi %$t %$v_o %$v_d %c_$t_0 %$v_f\n"
            "%$v_out_1 = OpPhi %bool %$v_s %$v_d %c_bool_0 %$v_f\n",
            v, value, v, v, value, v,
            v, v, v, v);
    }

    v->consumed = 1;
}
