/**
 * This file contains the logic that assembles a SPIR-V representation (in
 * textual assembly form) from a recorded Dr.Jit computation graph. It
 * implements a small template engine involving plentiful use of the 'fmt'
 * formatting routine.
 *
 * Its format interface supports the following format string characters.
 *
 *  Format  Input          Example result    Description
 * --------------------------------------------------------------------------
 *  $u      uint32_t      `1234`             Decimal number (32 bit)
 * --------------------------------------------------------------------------
 *  $s      const char *  `foo`              Zero-terminated string
 * --------------------------------------------------------------------------
 *  $t      Variable      `f32`              Variable type
 * --------------------------------------------------------------------------
 *  $b      Variable      `u32`              Variable type, binary format
 * --------------------------------------------------------------------------
 *  $v      Variable      `r1234`            Variable name
 * --------------------------------------------------------------------------
 *  $l      Variable      `0x1`              Literal value of variable (hex)
 * --------------------------------------------------------------------------
 *
 * SPIR-V identifiers carry a '%' prefix that the format strings add, e.g.
 * ``%$v = OpIAdd %$t %$v %$v``. Types, constants, and global variables are
 * declared on first use via ``fmt_intrinsic()``, which moves them into the
 * preamble of the unit (between the fixed prologue and the function body).
 */

#include "eval.h"
#include "internal.h"
#include "var.h"
#include "log.h"
#include "strbuf.h"
#include "vulkan_eval.h"
#include "vulkan_scatter.h"
#include "vulkan_packet.h"
#include "vulkan_array.h"
#include "array.h"
#include "vulkan.h"
#include "loop.h"
#include "cond.h"
#include "call.h"
#include <string>

// Forward declaration
static void jitc_vulkan_render(Variable *v);

void jitc_vulkan_declare_type(VarType vt) {
    switch (vt) {
        case VarType::Void:    fmt_intrinsic("%void = OpTypeVoid"); break;
        case VarType::Bool:    fmt_intrinsic("%bool = OpTypeBool"); break;
        case VarType::UInt8:   fmt_intrinsic("%u8 = OpTypeInt 8 0"); break;
        case VarType::UInt16:  fmt_intrinsic("%u16 = OpTypeInt 16 0"); break;
        case VarType::UInt32:  fmt_intrinsic("%u32 = OpTypeInt 32 0"); break;
        case VarType::UInt64:
        case VarType::Pointer: fmt_intrinsic("%u64 = OpTypeInt 64 0"); break;

        // Literals and bit-level operations of signed integer and floating
        // point values use the unsigned integer type of the same size
        case VarType::Int8:
            fmt_intrinsic("%u8 = OpTypeInt 8 0");
            fmt_intrinsic("%i8 = OpTypeInt 8 1");
            break;
        case VarType::Int16:
            fmt_intrinsic("%u16 = OpTypeInt 16 0");
            fmt_intrinsic("%i16 = OpTypeInt 16 1");
            break;
        case VarType::Int32:
            fmt_intrinsic("%u32 = OpTypeInt 32 0");
            fmt_intrinsic("%i32 = OpTypeInt 32 1");
            break;
        case VarType::Int64:
            fmt_intrinsic("%u64 = OpTypeInt 64 0");
            fmt_intrinsic("%i64 = OpTypeInt 64 1");
            break;
        case VarType::Float16:
            fmt_intrinsic("%u16 = OpTypeInt 16 0");
            fmt_intrinsic("%f16 = OpTypeFloat 16");
            break;
        case VarType::Float32:
            fmt_intrinsic("%u32 = OpTypeInt 32 0");
            fmt_intrinsic("%f32 = OpTypeFloat 32");
            break;
        case VarType::Float64:
            fmt_intrinsic("%u64 = OpTypeInt 64 0");
            fmt_intrinsic("%f64 = OpTypeFloat 64");
            break;

        default:
            jitc_raise("jitc_vulkan_declare_type(): type \"%s\" is not yet "
                       "supported by the Vulkan backend.", type_name[(int) vt]);
    }
}

void jitc_vulkan_declare_ptr_type(VarType vt) {
    jitc_vulkan_declare_type(vt);
    const char *name = type_name_spirv[(int) vt];
    fmt_intrinsic("%ptr_$s = OpTypePointer PhysicalStorageBuffer %$s",
                  name, name);
}

void jitc_vulkan_declare_const(VarType vt, uint32_t value) {
    jitc_vulkan_declare_type(vt);
    const char *name = type_name_spirv[(int) vt];
    if (vt == VarType::Bool)
        fmt_intrinsic("%c_bool_$u = $s %bool", value,
                      value ? "OpConstantTrue" : "OpConstantFalse");
    else
        fmt_intrinsic("%c_$s_$u = OpConstant %$s $u", name, value, name, value);
}

VarType jitc_vulkan_bin_type(VarType vt) {
    switch (type_size[(int) vt]) {
        case 1:  return VarType::UInt8;
        case 2:  return VarType::UInt16;
        case 4:  return VarType::UInt32;
        default: return VarType::UInt64;
    }
}

/// Load the entry of ``v`` in the kernel parameter array into '%<v>_pd'
static void jitc_vulkan_render_param(const Variable *v) {
    jitc_vulkan_declare_const(VarType::UInt64, v->param_offset);
    fmt("%$v_ps = OpIAdd %u64 %params %c_u64_$u\n"
        "%$v_pp = OpConvertUToPtr %ptr_u64 %$v_ps\n"
        "%$v_pd = OpLoad %u64 %$v_pp Aligned 8\n",
        v, v->param_offset, v, v, v, v);
}

void jitc_vulkan_render_element_ptr(const Variable *v, const char *tag,
                                    VarType vt, uint32_t base,
                                    const char *suffix, uint32_t index) {
    if (vt == VarType::Bool)
        vt = VarType::UInt8;
    jitc_vulkan_declare_ptr_type(vt);
    uint32_t isize = type_size[(int) vt];
    jitc_vulkan_declare_const(VarType::UInt64, isize);
    fmt("%$v_$si = OpUConvert %u64 %r$u\n"
        "%$v_$so = OpIMul %u64 %$v_$si %c_u64_$u\n"
        "%$v_$sa = OpIAdd %u64 %r$u$s %$v_$so\n"
        "%$v_$sp = OpConvertUToPtr %ptr_$s %$v_$sa\n",
        v, tag, index,
        v, tag, v, tag, isize,
        v, tag, base, suffix, v, tag,
        v, tag, type_name_spirv[(int) vt], v, tag);
}

/// Compute a pointer '%<v>_ep' to the element of the kernel input/output ``v``
/// processed by this thread
static void jitc_vulkan_render_param_ptr(const Variable *v) {
    if (v->is_array())
        jitc_raise("jitc_vulkan_render_param_ptr(): memory accesses to "
                   "variable r%u (type=%s) are not yet supported by the "
                   "Vulkan backend.", v->reg_index, type_name[v->type]);

    VarType vt = (VarType) v->type;
    if (v->size > 1) {
        jitc_vulkan_render_element_ptr(v, "e", vt, v->reg_index, "_pd", 0);
    } else {
        if (vt == VarType::Bool)
            vt = VarType::UInt8;
        jitc_vulkan_declare_ptr_type(vt);
        fmt("%$v_ep = OpConvertUToPtr %ptr_$s %$v_pd\n", v,
            type_name_spirv[(int) vt], v);
    }
}

void jitc_vulkan_render_load(const Variable *v, const char *tag,
                             const char *suffix) {
    if (jitc_is_bool(v)) {
        // 8-bit storage only permits loads, stores, and conversions
        jitc_vulkan_declare_const(VarType::UInt32, 0);
        fmt("%$v_$sm = OpLoad %u8 %$v_$sp Aligned 1\n"
            "%$v_$sw = OpUConvert %u32 %$v_$sm\n"
            "%$v$s = OpINotEqual %bool %$v_$sw %c_u32_0\n",
            v, tag, v, tag,
            v, tag, v, tag,
            v, suffix, v, tag);
    } else {
        fmt("%$v$s = OpLoad %$t %$v_$sp Aligned $u\n", v, suffix, v, v, tag,
            (uint32_t) type_size[v->type]);
    }
}

void jitc_vulkan_render_store(const Variable *v, const char *tag,
                              const Variable *value) {
    if (jitc_is_bool(value)) {
        jitc_vulkan_declare_const(VarType::UInt32, 0);
        jitc_vulkan_declare_const(VarType::UInt32, 1);
        fmt("%$v_$sw = OpSelect %u32 %$v %c_u32_1 %c_u32_0\n"
            "%$v_$sm = OpUConvert %u8 %$v_$sw\n"
            "OpStore %$v_$sp %$v_$sm Aligned 1\n",
            v, tag, value,
            v, tag, v, tag,
            v, tag, v, tag);
    } else {
        fmt("OpStore %$v_$sp %$v Aligned $u\n", v, tag, value,
            (uint32_t) type_size[value->type]);
    }
}

void jitc_vulkan_render_if(const Variable *v, const Variable *mask,
                           const char *suffix) {
    fmt("OpSelectionMerge %$v_e None\n"
        "OpBranchConditional %$v$s %$v_t %$v_f\n"
        "%$v_t = OpLabel\n",
        v, mask, suffix, v, v, v);
}

void jitc_vulkan_render_endif(const Variable *v) {
    fmt("OpBranch %$v_d\n"
        "%$v_d = OpLabel\n"
        "OpBranch %$v_e\n"
        "%$v_f = OpLabel\n"
        "OpBranch %$v_e\n"
        "%$v_e = OpLabel\n",
        v, v, v, v, v, v);
}

void jitc_vulkan_render_prologue(ThreadState *ts) {
    const VulkanDevice &dev = state.vulkan_devices[ts->device];

    put("OpCapability Shader\n"
        "OpCapability Int8\n"
        "OpCapability Int16\n"
        "OpCapability Int64\n"
        "OpCapability Float16\n"
        "OpCapability Float64\n"
        "OpCapability PhysicalStorageBufferAddresses\n"
        "OpCapability StorageBuffer8BitAccess\n"
        "OpCapability StorageBuffer16BitAccess\n"
        "OpCapability Int64Atomics\n"
        "OpCapability Sampled1D\n"
        "OpCapability RuntimeDescriptorArray\n"
        "OpCapability ShaderNonUniform\n"
        "OpCapability SampledImageArrayNonUniformIndexing\n");
    if (dev.atomic_float32_add)
        put("OpCapability AtomicFloat32AddEXT\n");
    if (dev.atomic_float64_add)
        put("OpCapability AtomicFloat64AddEXT\n");
    if (dev.atomic_float32_add || dev.atomic_float64_add)
        put("OpExtension \"SPV_EXT_shader_atomic_float_add\"\n");
    put("%glsl = OpExtInstImport \"GLSL.std.450\"\n"
        "OpMemoryModel PhysicalStorageBuffer64 GLSL450\n"
        "OpEntryPoint GLCompute %main "
        "\"drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\" %gid %nwg %push "
        "%tex1 %tex2 %tex3\n");
    fmt("OpExecutionMode %main LocalSize $u 1 1\n", dev.block_size);
    put("OpDecorate %gid BuiltIn GlobalInvocationId\n"
        "OpDecorate %nwg BuiltIn NumWorkgroups\n"
        "OpDecorate %Push Block\n"
        "OpMemberDecorate %Push 0 Offset 0\n");
    for (uint32_t i = 1; i <= 3; ++i)
        fmt("OpDecorate %tex$u DescriptorSet 0\n"
            "OpDecorate %tex$u Binding $u\n", i, i, i - 1);
}

void jitc_vulkan_assemble(ThreadState *ts, ScheduledGroup group,
                          uint32_t /*n_regs*/, uint32_t /*n_params*/) {
    // The push constant holds the address of the parameter array (size, followed
    // by the kernel parameters, see jitc_assemble())
    jitc_vulkan_declare_type(VarType::Void);
    jitc_vulkan_declare_type(VarType::Bool);
    jitc_vulkan_declare_ptr_type(VarType::UInt32);
    jitc_vulkan_declare_ptr_type(VarType::UInt64);
    fmt_intrinsic("%fn_void = OpTypeFunction %void");
    fmt_intrinsic("%v3u32 = OpTypeVector %u32 3");
    fmt_intrinsic("%ptr_in_v3u32 = OpTypePointer Input %v3u32");
    fmt_intrinsic("%gid = OpVariable %ptr_in_v3u32 Input");
    fmt_intrinsic("%nwg = OpVariable %ptr_in_v3u32 Input");
    fmt_intrinsic("%Push = OpTypeStruct %u64");
    fmt_intrinsic("%ptr_push = OpTypePointer PushConstant %Push");
    fmt_intrinsic("%ptr_push_u64 = OpTypePointer PushConstant %u64");
    fmt_intrinsic("%push = OpVariable %ptr_push PushConstant");
    jitc_vulkan_declare_const(VarType::UInt32, 0);
    const VulkanDevice &dev = state.vulkan_devices[ts->device];
    jitc_vulkan_declare_const(VarType::UInt32, dev.block_size);

    // The texture heap '%tex<ndim>' is part of the interface of every kernel
    // (see jitc_vulkan_render_prologue())
    jitc_vulkan_declare_type(VarType::Float32);
    for (uint32_t i = 1; i <= 3; ++i) {
        fmt_intrinsic("%img$u = OpTypeImage %f32 $s 0 0 0 1 Unknown", i,
                      i == 1 ? "1D" : (i == 2 ? "2D" : "3D"));
        fmt_intrinsic("%simg$u = OpTypeSampledImage %img$u", i, i);
        fmt_intrinsic("%ptr_simg$u = OpTypePointer UniformConstant %simg$u",
                      i, i);
        fmt_intrinsic("%arr_simg$u = OpTypeRuntimeArray %simg$u", i, i);
        fmt_intrinsic("%ptr_arr_simg$u = OpTypePointer UniformConstant "
                      "%arr_simg$u", i, i);
        fmt_intrinsic("%tex$u = OpVariable %ptr_arr_simg$u UniformConstant",
                      i, i);
    }

    // The number of workgroups is limited by the device, hence threads
    // process the elements in a grid-stride loop
    put("%main = OpFunction %void None %fn_void\n"
        "%entry = OpLabel\n");

    // Local arrays are declared here (see jitc_vulkan_render_array_alloc())
    jitc_vulkan_locals_target = buffer.size();

    fmt("%gid_v = OpLoad %v3u32 %gid\n"
        "%r0_start = OpCompositeExtract %u32 %gid_v 0\n"
        "%nwg_v = OpLoad %v3u32 %nwg\n"
        "%nwg_x = OpCompositeExtract %u32 %nwg_v 0\n"
        "%stride = OpIMul %u32 %nwg_x %c_u32_$u\n"
        "%params_p = OpAccessChain %ptr_push_u64 %push %c_u32_0\n"
        "%params = OpLoad %u64 %params_p\n"
        "%size_p = OpConvertUToPtr %ptr_u32 %params\n"
        "%size = OpLoad %u32 %size_p Aligned 8\n", dev.block_size);

    // Load the call-data base pointer once, shared by every call's dispatch
    if (call_buffer.base_v) {
        Variable *base = jitc_var(call_buffer.base_v);
        jitc_vulkan_render_param(base);
        fmt("%$v = OpCopyObject %u64 %$v_pd\n", base, base);
    }

    put("OpBranch %head\n"
        "%head = OpLabel\n"
        "%r0 = OpPhi %u32 %r0_start %entry %r0_next %next\n"
        "%active = OpULessThan %bool %r0 %size\n"
        "OpLoopMerge %done %next None\n"
        "OpBranchConditional %active %body %done\n"
        "%body = OpLabel\n");

    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        Variable *v = jitc_var(index);
        ParamType ptype = (ParamType) v->param_type;

        jitc_vulkan_declare_type((VarType) v->type);

        if (ptype == ParamType::Input) {
            if (v->is_literal()) {
                // Pointer literals must be loaded from the parameter array
                // (not inlined) so frozen function replay can update them
                if ((VarType) v->type == VarType::Pointer) {
                    jitc_vulkan_render_param(v);
                    fmt("%$v = OpCopyObject %u64 %$v_pd\n", v, v);
                } else {
                    jitc_vulkan_render(v);
                }
                continue;
            }

            jitc_vulkan_render_param(v);
            if (v->is_array()) {
                jitc_vulkan_render_array_memcpy_in(v);
            } else {
                jitc_vulkan_render_param_ptr(v);
                jitc_vulkan_render_load(v, "e", "");
            }
            continue;
        }

        jitc_vulkan_render(v);

        if (ptype == ParamType::Output) {
            jitc_vulkan_render_param(v);
            if (v->is_array()) {
                jitc_vulkan_render_array_memcpy_out(v);
            } else {
                jitc_vulkan_render_param_ptr(v);
                jitc_vulkan_render_store(v, "e", v);
            }
        }
    }

    // Step by at most 'size - r0' so that the index cannot overflow
    put("OpBranch %next\n"
        "%next = OpLabel\n"
        "%rest = OpISub %u32 %size %r0\n"
        "%step = OpExtInst %u32 %glsl UMin %stride %rest\n"
        "%r0_next = OpIAdd %u32 %r0 %step\n"
        "OpBranch %head\n"
        "%done = OpLabel\n"
        "OpReturn\n"
        "OpFunctionEnd\n");
}

/// Bit pattern of a quiet NaN, for use with the binary type '%$b'
static inline const char *jitc_vulkan_nan_str(const Variable *v) {
    if (jitc_is_half(v))
        return "0x7e00";
    else if (jitc_is_single(v))
        return "0x7fc00000";
    else
        return "0x7ff8000000000000";
}

void jitc_vulkan_render_unhandled(const Variable *v) {
    jitc_raise("jitc_vulkan_render(): unhandled VarKind=%u (%s) for "
               "variable r%u (type=%s).",
               (uint32_t) v->kind, var_kind_name[v->kind], v->reg_index,
               type_name[v->type]);
}

/// Emit a binary operation whose opcode depends on whether the operands are
/// floating point values, signed, or unsigned integers (nullptr: unsupported)
static void jitc_vulkan_render_binary(Variable *v, const char *op_f,
                                      const char *op_s, const char *op_u) {
    Variable *a0 = jitc_var(v->dep[0]),
             *a1 = jitc_var(v->dep[1]);
    const char *op = jitc_is_float(a0) ? op_f
                   : jitc_is_sint(a0)  ? op_s : op_u;
    if (!op || jitc_is_bool(a0))
        jitc_vulkan_render_unhandled(v);
    fmt("%$v = $s %$t %$v %$v\n", v, op, v, a0, a1);
}

/// Emit a call to a function of the GLSL.std.450 extended instruction set
static void jitc_vulkan_render_ext(Variable *v, const char *fn,
                                   uint32_t n_args) {
    fmt("%$v = OpExtInst %$t %glsl $s", v, v, fn);
    for (uint32_t i = 0; i < n_args; ++i)
        fmt(" %$v", jitc_var(v->dep[i]));
    put('\n');
}

/// Emit a bit-wise operation, reinterpreting floating point operands as
/// unsigned integers
static void jitc_vulkan_render_bitwise(Variable *v, const char *op,
                                       const char *op_bool) {
    Variable *a0 = jitc_var(v->dep[0]),
             *a1 = jitc_var(v->dep[1]);
    if (jitc_is_bool(v))
        fmt("%$v = $s %bool %$v %$v\n", v, op_bool, a0, a1);
    else if (jitc_is_float(v))
        fmt("%$v_0 = OpBitcast %$b %$v\n"
            "%$v_1 = OpBitcast %$b %$v\n"
            "%$v_2 = $s %$b %$v_0 %$v_1\n"
            "%$v = OpBitcast %$t %$v_2\n",
            v, v, a0, v, v, a1, v, op, v, v, v, v, v, v);
    else
        fmt("%$v = $s %$t %$v %$v\n", v, op, v, a0, a1);
}

/// Emit the operand '%<v>', or a null constant of the type of ``like`` if ``v``
/// was not assigned a register
static void jitc_vulkan_render_operand(const Variable *v, const Variable *like) {
    if (v->reg_index) {
        fmt(" %$v", v);
    } else {
        fmt_intrinsic("%null_$t = OpConstantNull %$t", like, like);
        fmt(" %null_$t", like);
    }
}

/// Emit a half precision division, square root, or reciprocal in single
/// precision. This matches the CUDA backend, which lacks half precision
/// variants of these operations (see jitc_fp16_min_compute_cuda()).
static void jitc_vulkan_render_half_as_single(Variable *v) {
    Variable *a0 = jitc_var(v->dep[0]);
    jitc_vulkan_declare_type(VarType::Float32);
    fmt("%$v_0 = OpFConvert %f32 %$v\n", v, a0);

    switch ((VarKind) v->kind) {
        case VarKind::Div:
        case VarKind::DivApprox:
            fmt("%$v_1 = OpFConvert %f32 %$v\n"
                "%$v_2 = OpFDiv %f32 %$v_0 %$v_1\n",
                v, jitc_var(v->dep[1]), v, v, v);
            break;

        case VarKind::Sqrt:
        case VarKind::SqrtApprox:
            fmt("%$v_2 = OpExtInst %f32 %glsl Sqrt %$v_0\n", v, v);
            break;

        default: // Rcp, RcpApprox
            jitc_vulkan_declare_const(VarType::Float32, 1);
            fmt("%$v_2 = OpFDiv %f32 %c_f32_1 %$v_0\n", v, v);
            break;
    }

    fmt("%$v = OpFConvert %f16 %$v_2\n", v, v);
}

/// Emit a rounding operation that is optionally fused with a float-to-int
/// conversion
static void jitc_vulkan_render_round(Variable *v, const char *fn) {
    Variable *a0 = jitc_var(v->dep[0]);
    const char *cvt = jitc_is_uint(v) ? "OpConvertFToU" : "OpConvertFToS";
    if (jitc_is_float(v))
        jitc_vulkan_render_ext(v, fn, 1);
    else if ((VarKind) v->kind == VarKind::Trunc)
        fmt("%$v = $s %$t %$v\n", v, cvt, v, a0);
    else
        fmt("%$v_r = OpExtInst %$t %glsl $s %$v\n"
            "%$v = $s %$t %$v_r\n",
            v, a0, fn, a0, v, cvt, v, v);
}

/// Emit a bit counting or reversal operation. Vulkan restricts these to 32-bit
/// operands, hence 64-bit values are processed as two halves, and 8/16-bit
/// values are zero-extended.
static void jitc_vulkan_render_bits(Variable *v) {
    Variable *a0 = jitc_var(v->dep[0]);
    VarKind kind = (VarKind) v->kind;
    uint32_t bits = type_size[v->type] * 8;
    bool wide = bits == 64, narrow = bits < 32;

    jitc_vulkan_declare_const(VarType::UInt32, 0);
    jitc_vulkan_declare_const(VarType::UInt32, 31);
    jitc_vulkan_declare_const(VarType::UInt32, 32);

    // Unsigned view '%<v>_x' of the operand (zero-extended to 32 bit)
    if (narrow)
        fmt("%$v_b = $s %$b %$v\n"
            "%$v_x = OpUConvert %u32 %$v_b\n", v,
            jitc_is_sint(v) ? "OpBitcast" : "OpCopyObject", v, a0, v, v);
    else
        fmt("%$v_x = $s %$b %$v\n", v,
            jitc_is_sint(v) ? "OpBitcast" : "OpCopyObject", v, a0);

    if (!wide) {
        switch (kind) {
            case VarKind::Popc:
                fmt("%$v_w = OpBitCount %u32 %$v_x\n", v, v);
                break;
            case VarKind::Clz:
                fmt("%$v_m = OpExtInst %u32 %glsl FindUMsb %$v_x\n"
                    "%$v_w = OpISub %u32 %c_u32_31 %$v_m\n", v, v, v, v);
                break;
            case VarKind::Ctz:
                fmt("%$v_l = OpExtInst %u32 %glsl FindILsb %$v_x\n"
                    "%$v_z = OpIEqual %bool %$v_x %c_u32_0\n"
                    "%$v_w = OpSelect %u32 %$v_z %c_u32_32 %$v_l\n",
                    v, v, v, v, v, v, v);
                break;
            default: // Brev
                fmt("%$v_w = OpBitReverse %u32 %$v_x\n", v, v);
                break;
        }
    } else {
        jitc_vulkan_declare_const(VarType::UInt64, 32);
        fmt("%$v_l = OpUConvert %u32 %$v_x\n"
            "%$v_s = OpShiftRightLogical %u64 %$v_x %c_u64_32\n"
            "%$v_h = OpUConvert %u32 %$v_s\n",
            v, v, v, v, v, v);

        switch (kind) {
            case VarKind::Popc:
                fmt("%$v_c = OpBitCount %u32 %$v_l\n"
                    "%$v_d = OpBitCount %u32 %$v_h\n"
                    "%$v_y = OpIAdd %u32 %$v_c %$v_d\n"
                    "%$v_w = OpUConvert %u64 %$v_y\n",
                    v, v, v, v, v, v, v, v, v);
                break;
            case VarKind::Clz:
                fmt("%$v_m = OpExtInst %u32 %glsl FindUMsb %$v_h\n"
                    "%$v_c = OpISub %u32 %c_u32_31 %$v_m\n"
                    "%$v_n = OpExtInst %u32 %glsl FindUMsb %$v_l\n"
                    "%$v_d = OpISub %u32 %c_u32_31 %$v_n\n"
                    "%$v_e = OpIAdd %u32 %c_u32_32 %$v_d\n"
                    "%$v_z = OpIEqual %bool %$v_h %c_u32_0\n"
                    "%$v_y = OpSelect %u32 %$v_z %$v_e %$v_c\n"
                    "%$v_w = OpUConvert %u64 %$v_y\n",
                    v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v);
                break;
            case VarKind::Ctz:
                fmt("%$v_c = OpExtInst %u32 %glsl FindILsb %$v_l\n"
                    "%$v_d = OpExtInst %u32 %glsl FindILsb %$v_h\n"
                    "%$v_q = OpIEqual %bool %$v_h %c_u32_0\n"
                    "%$v_f = OpSelect %u32 %$v_q %c_u32_32 %$v_d\n"
                    "%$v_e = OpIAdd %u32 %c_u32_32 %$v_f\n"
                    "%$v_z = OpIEqual %bool %$v_l %c_u32_0\n"
                    "%$v_y = OpSelect %u32 %$v_z %$v_e %$v_c\n"
                    "%$v_w = OpUConvert %u64 %$v_y\n",
                    v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v);
                break;
            default: // Brev
                fmt("%$v_c = OpBitReverse %u32 %$v_l\n"
                    "%$v_d = OpBitReverse %u32 %$v_h\n"
                    "%$v_f = OpUConvert %u64 %$v_c\n"
                    "%$v_g = OpShiftLeftLogical %u64 %$v_f %c_u64_32\n"
                    "%$v_k = OpUConvert %u64 %$v_d\n"
                    "%$v_w = OpBitwiseOr %u64 %$v_g %$v_k\n",
                    v, v, v, v, v, v, v, v, v, v, v, v, v);
                break;
        }
    }

    if (narrow) {
        // Remove the contribution of the zero-extension
        jitc_vulkan_declare_const(VarType::UInt32, bits);
        jitc_vulkan_declare_const(VarType::UInt32, 32 - bits);
        switch (kind) {
            case VarKind::Popc:
                fmt("%$v_a = OpCopyObject %u32 %$v_w\n", v, v);
                break;
            case VarKind::Clz:
                fmt("%$v_a = OpISub %u32 %$v_w %c_u32_$u\n", v, v, 32 - bits);
                break;
            case VarKind::Ctz:
                fmt("%$v_a = OpExtInst %u32 %glsl UMin %$v_w %c_u32_$u\n",
                    v, v, bits);
                break;
            default: // Brev
                fmt("%$v_a = OpShiftRightLogical %u32 %$v_w %c_u32_$u\n",
                    v, v, 32 - bits);
                break;
        }
        fmt("%$v_n = OpUConvert %$b %$v_a\n", v, v, v);
    }

    fmt("%$v = $s %$t %$v_$s\n", v,
        jitc_is_sint(v) ? "OpBitcast" : "OpCopyObject", v, v,
        narrow ? "n" : "w");
}

/// Sample the texture in the heap slot 'dep[0]' at the coordinates
/// 'dep[1..3]' into '%<v>_out_<i>'. Masked lookups (see
/// jitc_vulkan_tex_lookup()) are skipped in lanes whose slot is zero.
static void jitc_vulkan_render_tex_lookup(const Variable *v) {
    const Variable *slot = jitc_var(v->dep[0]),
                   *c0 = jitc_var(v->dep[1]),
                   *c1 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr,
                   *c2 = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;
    uint32_t ndim = c2 ? 3 : (c1 ? 2 : 1);
    bool masked = v->literal != 0;

    fmt_intrinsic("%v4f32 = OpTypeVector %f32 4");
    jitc_vulkan_declare_const(VarType::Float32, 0);

    if (masked) {
        jitc_vulkan_declare_const(VarType::UInt64, 0);
        fmt("%$v_m = OpINotEqual %bool %$v %c_u64_0\n", v, slot);
        jitc_vulkan_render_if(v, v, "_m");
    }

    // The slot may differ between lanes
    fmt("%$v_ti = OpUConvert %u32 %$v\n"
        "%$v_tp = OpAccessChain %ptr_simg$u %tex$u %$v_ti\n"
        "%$v_ts = OpLoad %simg$u %$v_tp\n"
        "OpDecorate %$v_tp NonUniform\n"
        "OpDecorate %$v_ts NonUniform\n",
        v, slot,
        v, ndim, ndim, v,
        v, ndim, v,
        v, v);

    if (ndim == 1) {
        fmt("%$v_tc = OpCopyObject %f32 %$v\n", v, c0);
    } else if (ndim == 2) {
        fmt_intrinsic("%v2f32 = OpTypeVector %f32 2");
        fmt("%$v_tc = OpCompositeConstruct %v2f32 %$v %$v\n", v, c0, c1);
    } else {
        fmt_intrinsic("%v3f32 = OpTypeVector %f32 3");
        fmt("%$v_tc = OpCompositeConstruct %v3f32 %$v %$v %$v\n", v, c0, c1,
            c2);
    }

    fmt("%$v_s = OpImageSampleExplicitLod %v4f32 %$v_ts %$v_tc Lod "
        "%c_f32_0\n", v, v, v);

    if (masked) {
        fmt_intrinsic("%null_v4f32 = OpConstantNull %v4f32");
        jitc_vulkan_render_endif(v);
        fmt("%$v_r = OpPhi %v4f32 %$v_s %$v_d %null_v4f32 %$v_f\n", v, v, v,
            v);
    }

    for (uint32_t i = 0; i < 4; ++i)
        fmt("%$v_out_$u = OpCompositeExtract %f32 %$v_$s $u\n", v, i, v,
            masked ? "r" : "s", i);
}

static void jitc_vulkan_render(Variable *v) {
    Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
             *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
             *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr;
    VarType vt = (VarType) v->type;

    switch ((VarKind) v->kind) {
        case VarKind::Nop:
            break;

        case VarKind::Undefined:
        case VarKind::Literal:
            // Constants are named after their value, so that the body of a
            // callable (which determines its hash) references the value, and
            // the declarations of different units coincide
            if (vt == VarType::Bool) {
                jitc_vulkan_declare_const(vt, v->literal ? 1 : 0);
                fmt("%$v = OpCopyObject %bool %c_bool_$u\n", v,
                    v->literal ? 1 : 0);
            } else {
                // Signed and floating point constants are bit patterns of the
                // corresponding unsigned type
                fmt_intrinsic("%c_$b_$l = OpConstant %$b $l", v, v, v, v);
                fmt("%$v = $s %$t %c_$b_$l\n", v,
                    strcmp(type_name_spirv[v->type],
                           type_name_spirv_bin[v->type]) == 0
                        ? "OpCopyObject" : "OpBitcast",
                    v, v, v);
            }
            break;

        case VarKind::Counter:
            fmt("%$v = OpCopyObject %$t %r0\n", v, v);
            break;

        // -- Bounds check: record the first offending index of an active lane
        // in the buffer 'a2', and disable the lane --
        case VarKind::BoundsCheck: {
            uint32_t size = (uint32_t) v->literal;
            jitc_vulkan_declare_const(VarType::UInt32, size);
            fmt("%$v_o = OpUGreaterThanEqual %bool %$v %c_u32_$u\n"
                "%$v_c = OpLogicalAnd %bool %$v_o %$v\n"
                "OpSelectionMerge %$v_e None\n"
                "OpBranchConditional %$v_c %$v_t %$v_e\n"
                "%$v_t = OpLabel\n"
                "%$v_p = OpConvertUToPtr %ptr_u32 %$v\n"
                "OpStore %$v_p %$v Aligned 4\n"
                "OpBranch %$v_e\n"
                "%$v_e = OpLabel\n"
                "%$v = OpLogicalNotEqual %bool %$v_c %$v\n",
                v, a0, size,
                v, v, a1,
                v,
                v, v, v,
                v,
                v, a2,
                v, a0,
                v,
                v,
                v, v, a1);
            break;
        }

        // -- Memory operations --
        case VarKind::Gather: {
            bool is_unmasked = a2->is_literal() && a2->literal == 1;
            jitc_vulkan_render_element_ptr(v, "g", vt, a0->reg_index, "",
                                           a1->reg_index);
            if (is_unmasked) {
                jitc_vulkan_render_load(v, "g", "");
            } else {
                // Masked lanes skip the load (their index may be invalid) and
                // produce zero
                jitc_vulkan_declare_const(vt, 0);
                jitc_vulkan_render_if(v, a2);
                jitc_vulkan_render_load(v, "g", "_l");
                jitc_vulkan_render_endif(v);
                fmt("%$v = OpPhi %$t %$v_l %$v_d %c_$t_0 %$v_f\n",
                    v, v, v, v, v, v);
            }
            break;
        }

        case VarKind::PacketGather:
            jitc_vulkan_render_gather_packet(v, a0, a1, a2);
            break;

        case VarKind::PacketScatter:
            jitc_vulkan_render_scatter_packet(v, a0, a1, a2);
            break;

        case VarKind::Scatter:     jitc_vulkan_render_scatter(v);      break;
        case VarKind::ScatterInc:  jitc_vulkan_render_scatter_inc(v);  break;
        case VarKind::ScatterExch: jitc_vulkan_render_scatter_exch(v); break;
        case VarKind::ScatterCAS:  jitc_vulkan_render_scatter_cas(v);  break;

        case VarKind::TexLookup:
            jitc_vulkan_render_tex_lookup(v);
            break;

        // Outputs of operations with several results are '%<src>_out_<i>'
        case VarKind::Extract:
            fmt("%$v = OpCopyObject %$t %$v_out_$u\n", v, v, a0,
                (uint32_t) v->literal);
            break;

        // -- Arithmetic --
        case VarKind::Add: jitc_vulkan_render_binary(v, "OpFAdd", "OpIAdd", "OpIAdd"); break;
        case VarKind::Sub: jitc_vulkan_render_binary(v, "OpFSub", "OpISub", "OpISub"); break;
        case VarKind::Mul: jitc_vulkan_render_binary(v, "OpFMul", "OpIMul", "OpIMul"); break;
        case VarKind::Div:
        case VarKind::DivApprox:
            if (jitc_is_half(v))
                jitc_vulkan_render_half_as_single(v);
            else
                jitc_vulkan_render_binary(v, "OpFDiv", "OpSDiv", "OpUDiv");
            break;
        // NVIDIA's driver (615.71) evaluates OpSRem like OpUMod, hence the
        // remainder is computed from the quotient
        case VarKind::Mod:
            if (jitc_is_sint(v))
                fmt("%$v_q = OpSDiv %$t %$v %$v\n"
                    "%$v_m = OpIMul %$t %$v_q %$v\n"
                    "%$v = OpISub %$t %$v %$v_m\n",
                    v, v, a0, a1, v, v, v, a1, v, v, a0, v);
            else
                jitc_vulkan_render_binary(v, nullptr, nullptr, "OpUMod");
            break;

        case VarKind::Fma:
            if (jitc_is_float(v))
                jitc_vulkan_render_ext(v, "Fma", 3);
            else
                fmt("%$v_m = OpIMul %$t %$v %$v\n"
                    "%$v = OpIAdd %$t %$v_m %$v\n",
                    v, v, a0, a1, v, v, v, a2);
            break;

        // GLSL's FMin/FMax are undefined for NaN arguments. Selecting a NaN
        // propagates them like LLVM's llvm.minimum() and Metal.
        case VarKind::Min:
        case VarKind::Max: {
            bool is_min = (VarKind) v->kind == VarKind::Min;
            if (jitc_is_float(v)) {
                fmt_intrinsic("%nan_$b = OpConstant %$b $s", v, v,
                              jitc_vulkan_nan_str(v));
                fmt("%$v_m = OpExtInst %$t %glsl $s %$v %$v\n"
                    "%$v_a = OpIsNan %bool %$v\n"
                    "%$v_b = OpIsNan %bool %$v\n"
                    "%$v_o = OpLogicalOr %bool %$v_a %$v_b\n"
                    "%$v_n = OpBitcast %$t %nan_$b\n"
                    "%$v = OpSelect %$t %$v_o %$v_n %$v_m\n",
                    v, v, is_min ? "FMin" : "FMax", a0, a1,
                    v, a0,
                    v, a1,
                    v, v, v,
                    v, v, v,
                    v, v, v, v, v);
            } else if (jitc_is_int(v)) {
                jitc_vulkan_render_ext(
                    v, jitc_is_sint(v) ? (is_min ? "SMin" : "SMax")
                                       : (is_min ? "UMin" : "UMax"), 2);
            } else {
                jitc_vulkan_render_unhandled(v);
            }
            break;
        }

        case VarKind::FMin: jitc_vulkan_render_ext(v, "NMin", 2); break;
        case VarKind::FMax: jitc_vulkan_render_ext(v, "NMax", 2); break;

        case VarKind::MulHi:
            fmt_intrinsic("%mulx_$t = OpTypeStruct %$t %$t", v, v, v);
            fmt("%$v_x = $s %mulx_$t %$v %$v\n"
                "%$v = OpCompositeExtract %$t %$v_x 1\n",
                v, jitc_is_sint(v) ? "OpSMulExtended" : "OpUMulExtended", v,
                a0, a1, v, v, v);
            break;

        case VarKind::MulWide: {
            // OpUConvert requires an unsigned result type (the case for
            // unsigned operands)
            const char *cvt = jitc_is_sint(a0) ? "OpSConvert" : "OpUConvert";
            fmt("%$v_0 = $s %$t %$v\n"
                "%$v_1 = $s %$t %$v\n"
                "%$v = OpIMul %$t %$v_0 %$v_1\n",
                v, cvt, v, a0, v, cvt, v, a1, v, v, v, v);
            break;
        }

        case VarKind::Copysign: {
            // Take the sign bit from a1 and everything else from a0
            const char *mask;
            if (jitc_is_half(v))
                mask = "0x8000";
            else if (jitc_is_single(v))
                mask = "0x80000000";
            else
                mask = "0x8000000000000000";
            fmt_intrinsic("%sign_$b = OpConstant %$b $s", v, v, mask);
            fmt("%$v_0 = OpBitcast %$b %$v\n"
                "%$v_1 = OpBitcast %$b %$v\n"
                "%$v_2 = OpBitwiseXor %$b %$v_0 %$v_1\n"
                "%$v_3 = OpBitwiseAnd %$b %$v_2 %sign_$b\n"
                "%$v_4 = OpBitwiseXor %$b %$v_0 %$v_3\n"
                "%$v = OpBitcast %$t %$v_4\n",
                v, v, a0,
                v, v, a1,
                v, v, v, v,
                v, v, v, v,
                v, v, v, v,
                v, v, v);
            break;
        }

        // -- Rounding --
        case VarKind::Ceil:  jitc_vulkan_render_round(v, "Ceil");      break;
        case VarKind::Floor: jitc_vulkan_render_round(v, "Floor");     break;
        case VarKind::Round: jitc_vulkan_render_round(v, "RoundEven"); break;
        case VarKind::Trunc: jitc_vulkan_render_round(v, "Trunc");     break;

        // -- Comparisons --
        case VarKind::Eq:
            if (jitc_is_bool(a0))
                fmt("%$v = OpLogicalEqual %bool %$v %$v\n", v, a0, a1);
            else
                jitc_vulkan_render_binary(v, "OpFOrdEqual", "OpIEqual", "OpIEqual");
            break;

        case VarKind::Neq:
            if (jitc_is_bool(a0))
                fmt("%$v = OpLogicalNotEqual %bool %$v %$v\n", v, a0, a1);
            else
                jitc_vulkan_render_binary(v, "OpFOrdNotEqual", "OpINotEqual", "OpINotEqual");
            break;

        case VarKind::Lt:
        case VarKind::Le:
        case VarKind::Gt:
        case VarKind::Ge: {
            if (jitc_is_bool(a0))
                jitc_vulkan_render_unhandled(v);
            static const char *ops[4][3] = {
                { "OpFOrdLessThan", "OpSLessThan", "OpULessThan" },
                { "OpFOrdLessThanEqual", "OpSLessThanEqual", "OpULessThanEqual" },
                { "OpFOrdGreaterThan", "OpSGreaterThan", "OpUGreaterThan" },
                { "OpFOrdGreaterThanEqual", "OpSGreaterThanEqual", "OpUGreaterThanEqual" }
            };
            const char **op = ops[v->kind - (uint32_t) VarKind::Lt];
            jitc_vulkan_render_binary(v, op[0], op[1], op[2]);
            break;
        }

        // -- Ternary --
        case VarKind::Select:
            fmt("%$v = OpSelect %$t %$v %$v %$v\n", v, v, a0, a1, a2);
            break;

        // -- Bit operations --
        case VarKind::And:
            if (a0->type != a1->type) {
                jitc_vulkan_declare_const(vt, 0);
                fmt("%$v = OpSelect %$t %$v %$v %c_$t_0\n", v, v, a1, a0, v);
            } else {
                jitc_vulkan_render_bitwise(v, "OpBitwiseAnd", "OpLogicalAnd");
            }
            break;

        case VarKind::Or:
            if (a0->type != a1->type) {
                if (jitc_is_float(v)) {
                    jitc_vulkan_declare_const(jitc_vulkan_bin_type(vt), 0);
                    fmt("%$v_0 = OpBitcast %$b %$v\n"
                        "%$v_1 = OpNot %$b %c_$b_0\n"
                        "%$v_2 = OpSelect %$b %$v %$v_1 %$v_0\n"
                        "%$v = OpBitcast %$t %$v_2\n",
                        v, v, a0, v, v, v, v, v, a1, v, v, v, v, v);
                } else {
                    jitc_vulkan_declare_const(vt, 0);
                    fmt("%$v_1 = OpNot %$t %c_$t_0\n"
                        "%$v = OpSelect %$t %$v %$v_1 %$v\n",
                        v, v, v, v, v, a1, v, a0);
                }
            } else {
                jitc_vulkan_render_bitwise(v, "OpBitwiseOr", "OpLogicalOr");
            }
            break;

        case VarKind::Xor:
            jitc_vulkan_render_bitwise(v, "OpBitwiseXor", "OpLogicalNotEqual");
            break;

        case VarKind::Shl:
            jitc_vulkan_render_binary(v, nullptr, "OpShiftLeftLogical",
                                      "OpShiftLeftLogical");
            break;

        case VarKind::Shr:
            jitc_vulkan_render_binary(v, nullptr, "OpShiftRightArithmetic",
                                      "OpShiftRightLogical");
            break;

        case VarKind::Popc:
        case VarKind::Clz:
        case VarKind::Ctz:
        case VarKind::Brev:
            jitc_vulkan_render_bits(v);
            break;

        // -- Unary --
        case VarKind::Neg:
            fmt("%$v = $s %$t %$v\n", v,
                jitc_is_float(v) ? "OpFNegate" : "OpSNegate", v, a0);
            break;

        case VarKind::Not:
            if (jitc_is_bool(v))
                fmt("%$v = OpLogicalNot %bool %$v\n", v, a0);
            else if (jitc_is_float(v))
                fmt("%$v_0 = OpBitcast %$b %$v\n"
                    "%$v_1 = OpNot %$b %$v_0\n"
                    "%$v = OpBitcast %$t %$v_1\n",
                    v, v, a0, v, v, v, v, v, v);
            else
                fmt("%$v = OpNot %$t %$v\n", v, v, a0);
            break;

        // LLVM's llvm.abs() also interprets unsigned integers as signed
        case VarKind::Abs:
            jitc_vulkan_render_ext(v, jitc_is_float(v) ? "FAbs" : "SAbs", 1);
            break;

        case VarKind::Sqrt:
        case VarKind::SqrtApprox:
            if (jitc_is_half(v))
                jitc_vulkan_render_half_as_single(v);
            else
                jitc_vulkan_render_ext(v, "Sqrt", 1);
            break;

        // -- Reciprocal --
        case VarKind::Rcp:
        case VarKind::RcpApprox:
            if (jitc_is_half(v)) {
                jitc_vulkan_render_half_as_single(v);
            } else {
                jitc_vulkan_declare_const(vt, 1);
                fmt("%$v = OpFDiv %$t %c_$t_1 %$v\n", v, v, v, a0);
            }
            break;

        case VarKind::RSqrtApprox: jitc_vulkan_render_ext(v, "InverseSqrt", 1); break;

        // -- Transcendental functions (approximations in Vulkan). GLSL.std.450
        // only provides them for 16- and 32-bit operands. --
        case VarKind::Sin:
        case VarKind::Cos:
        case VarKind::Exp2:
        case VarKind::Log2:
        case VarKind::Tanh: {
            if (jitc_is_double(v))
                jitc_vulkan_render_unhandled(v);
            static const char *fns[] = { "Sin", "Cos", "Exp2", "Log2", "Tanh" };
            jitc_vulkan_render_ext(v, fns[v->kind - (uint32_t) VarKind::Sin], 1);
            break;
        }

        // -- Casts --
        case VarKind::Cast: {
            const char *op;
            if (jitc_is_bool(v)) {
                jitc_vulkan_declare_const((VarType) a0->type, 0);
                fmt("%$v = $s %bool %$v %c_$t_0\n", v,
                    jitc_is_float(a0) ? "OpFOrdNotEqual" : "OpINotEqual",
                    a0, a0);
                break;
            } else if (jitc_is_bool(a0)) {
                jitc_vulkan_declare_const(vt, 0);
                jitc_vulkan_declare_const(vt, 1);
                fmt("%$v = OpSelect %$t %$v %c_$t_1 %c_$t_0\n",
                    v, v, a0, v, v);
                break;
            } else if (jitc_is_float(v) && jitc_is_float(a0)) {
                op = "OpFConvert";
            } else if (jitc_is_float(v)) {
                op = jitc_is_uint(a0) ? "OpConvertUToF" : "OpConvertSToF";
            } else if (jitc_is_float(a0)) {
                op = jitc_is_uint(v) ? "OpConvertFToU" : "OpConvertFToS";
            } else if (type_size[v->type] == type_size[a0->type]) {
                op = "OpBitcast";
            } else if (type_size[v->type] < type_size[a0->type] ||
                       jitc_is_sint(a0)) {
                op = "OpSConvert";
            } else if (jitc_is_uint(v)) {
                op = "OpUConvert";
            } else {
                // OpUConvert requires an unsigned result type
                fmt("%$v_u = OpUConvert %$b %$v\n"
                    "%$v = OpBitcast %$t %$v_u\n",
                    v, v, a0, v, v, v);
                break;
            }
            fmt("%$v = $s %$t %$v\n", v, op, v, a0);
            break;
        }

        case VarKind::Bitcast:
            fmt("%$v = $s %$t %$v\n", v,
                v->type == a0->type ? "OpCopyObject" : "OpBitcast", v, a0);
            break;

        // -- Loops. The loop state variables are phi nodes in the header block
        // '%<loop>_lh', whose predecessors are '%<loop>_lp' (loop entry) and
        // '%<loop>_lk' (back edge). The condition is evaluated in '%<loop>_lc'.
        case VarKind::LoopStart: {
            const LoopData *ld = (LoopData *) v->data;
            if (ld->name != "unnamed")
                fmt("; Symbolic loop: $s\n", ld->name.c_str());
            fmt("OpBranch %$v_lp\n"
                "%$v_lp = OpLabel\n"
                "OpBranch %$v_lh\n"
                "%$v_lh = OpLabel\n",
                v, v, v, v);
            for (size_t i = 0; i < ld->size; ++i) {
                Variable *inner_in  = jitc_var(ld->inner_in[i]),
                         *outer_in  = jitc_var(ld->outer_in[i]),
                         *inner_out = jitc_var(ld->inner_out[i]);
                // Ignore eliminated/unused loop state variables. Arrays are
                // modified in place and need no phi nodes.
                if (inner_in == outer_in || !inner_in->reg_index ||
                    inner_in->is_array())
                    continue;
                fmt("%$v = OpPhi %$t", inner_in, inner_in);
                jitc_vulkan_render_operand(outer_in, inner_in);
                fmt(" %$v_lp", v);
                jitc_vulkan_render_operand(inner_out->reg_index ? inner_out : inner_in,
                                           inner_in);
                fmt(" %$v_lk\n", v);
            }
            fmt("OpLoopMerge %$v_ld %$v_lk None\n"
                "OpBranch %$v_lc\n"
                "%$v_lc = OpLabel\n",
                v, v, v, v);
            break;
        }

        case VarKind::LoopCond:
            fmt("OpBranchConditional %$v %$v_lb %$v_ld\n"
                "%$v_lb = OpLabel\n",
                a1, a0, a0, a0);
            break;

        case VarKind::LoopEnd:
            fmt("OpBranch %$v_lk\n"
                "%$v_lk = OpLabel\n"
                "OpBranch %$v_lh\n"
                "%$v_ld = OpLabel\n",
                a0, a0, a0, a0);
            break;

        case VarKind::LoopPhi:
            // The phi node is emitted by LoopStart, arrays share the storage
            if (v->is_array())
                v->reg_index = jitc_var(v->dep[3])->reg_index;
            break;

        // The header block dominates the loop exit, hence the phi node holds
        // the final value of the loop state variable
        case VarKind::LoopOutput: {
            const LoopData *ld = (LoopData *) a0->data;
            for (size_t i = 0; i < ld->size; ++i) {
                if (jitc_var(ld->outer_out[i]) != v)
                    continue;
                Variable *inner_in = jitc_var(ld->inner_in[i]);
                if (v->is_array()) {
                    v->reg_index = inner_in->reg_index;
                } else {
                    fmt("%$v = OpCopyObject %$t", v, v);
                    jitc_vulkan_render_operand(inner_in, v);
                    put('\n');
                }
                break;
            }
            break;
        }

        // -- Conditionals. The outputs are phi nodes in the merge block
        // '%<cond>_ce', whose predecessors are '%<cond>_cu' (end of the 'true'
        // branch) and '%<cond>_cv' (end of the 'false' branch).
        case VarKind::CondStart: {
            const CondData *cd = (CondData *) v->data;
            if (cd->name != "unnamed")
                fmt("; Symbolic conditional: $s\n", cd->name.c_str());
            fmt("OpSelectionMerge %$v_ce None\n"
                "OpBranchConditional %$v %$v_ct %$v_cf\n"
                "%$v_ct = OpLabel\n",
                v, a0, v, v, v);
            break;
        }

        case VarKind::CondMid:
            fmt("OpBranch %$v_cu\n"
                "%$v_cu = OpLabel\n"
                "OpBranch %$v_ce\n"
                "%$v_cf = OpLabel\n",
                a0, a0, a0, a0);
            break;

        case VarKind::CondEnd: {
            const CondData *cd = (CondData *) a0->data;
            fmt("OpBranch %$v_cv\n"
                "%$v_cv = OpLabel\n"
                "OpBranch %$v_ce\n"
                "%$v_ce = OpLabel\n",
                a0, a0, a0, a0);
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vo = jitc_var(cd->indices_out[i]);
                if (!jitc_cond_output_active(vo))
                    continue;
                fmt("%$v = OpPhi %$t", vo, vo);
                jitc_vulkan_render_operand(jitc_var(cd->indices_t[i]), vo);
                fmt(" %$v_cu", a0);
                jitc_vulkan_render_operand(jitc_var(cd->indices_f[i]), vo);
                fmt(" %$v_cv\n", a0);
            }
            break;
        }

        case VarKind::CondOutput: // The phi node is emitted by CondEnd
            break;

        // -- Local arrays --
        case VarKind::Array:
            jitc_vulkan_render_array(v, v->dep[0] ? a0 : nullptr);
            break;

        case VarKind::ArrayInit:
            jitc_vulkan_render_array_init(v, a0, a1);
            break;

        case VarKind::ArrayWrite:
            jitc_vulkan_render_array_write(v, a0, a1, a2,
                                           v->dep[3] ? jitc_var(v->dep[3]) : nullptr);
            break;

        case VarKind::ArrayRead:
            jitc_vulkan_render_array_read(v, a0, a1, a2);
            break;

        case VarKind::ArrayPhi:
            v->reg_index = a0->reg_index;
            break;

        case VarKind::ArraySelect:
            jitc_vulkan_render_array_select(v, a0, a1, a2);
            break;

        // -- Function calls --
        case VarKind::Call:
            jitc_var_call_assemble((CallData *) v->data, v->reg_index,
                                   a0->reg_index, a1->reg_index);
            break;

        case VarKind::CallGetter:
            jitc_var_call_getter_assemble(v, a0, a1);
            break;

        case VarKind::CallOutput: // Defined by the dispatch of the call
            break;

        default:
            jitc_vulkan_render_unhandled(v);
    }
}

// ============================================================================
//  Function calls. SPIR-V has no function pointers, hence the dispatch is an
//  'OpSwitch' over the callable index. Its case literals are placeholders
//  '@<hash>' that jitc_vulkan_kernel_compile() replaces by the indices
//  assigned in jitc_unit_finalize(), when it combines the units into a module.
// ============================================================================

/// Declare the types of a call's callables: the struct '%ret_<types>' of its
/// live outputs (or '%void'), and the function type '%fn_<ret>_<arguments>'.
/// The arguments are 'index', 'self', 'data', 'base' (with nested calls), and
/// the active inputs.
static void jitc_vulkan_declare_call_types(const CallData *call,
                                           std::string &ret, std::string &fn) {
    std::string members;
    ret = "ret";
    for (uint32_t i = 0; i < call->n_out; ++i) {
        if (call->out_offset[i] == (uint32_t) -1)
            continue;
        VarType vt = (VarType) jitc_var(call->inner_out[i])->type;
        jitc_vulkan_declare_type(vt);
        ret += '_';
        ret += type_name_spirv[(int) vt];
        members += " %";
        members += type_name_spirv[(int) vt];
    }

    if (members.empty()) {
        ret = "void";
        jitc_vulkan_declare_type(VarType::Void);
    } else {
        fmt_intrinsic("%$s = OpTypeStruct$s", ret.c_str(), members.c_str());
        fmt_intrinsic("%null_$s = OpConstantNull %$s", ret.c_str(), ret.c_str());
    }

    jitc_vulkan_declare_type(VarType::UInt32);
    jitc_vulkan_declare_type(VarType::UInt64);
    fn = "fn_" + ret + "_u32_u32_u64";
    std::string params = " %u32 %u32 %u64";
    if (call->use_nested) {
        fn += "_u64";
        params += " %u64";
    }
    for (uint32_t i = 0; i < call->n_in; ++i) {
        if (!call->in_active[i])
            continue;
        VarType vt = (VarType) jitc_var(call->outer_in[i])->type;
        jitc_vulkan_declare_type(vt);
        fn += '_';
        fn += type_name_spirv[(int) vt];
        params += " %";
        params += type_name_spirv[(int) vt];
    }
    fmt_intrinsic("%$s = OpTypeFunction %$s$s", fn.c_str(), ret.c_str(),
                  params.c_str());
}

void jitc_vulkan_assemble_func(const CallData *call, uint32_t inst) {
    std::string ret, fn;
    jitc_vulkan_declare_call_types(call, ret, fn);

    fmt("%func_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ = OpFunction %$s None %$s\n"
        "%index = OpFunctionParameter %u32\n"
        "%self = OpFunctionParameter %u32\n"
        "%data = OpFunctionParameter %u64\n",
        ret.c_str(), fn.c_str());
    if (call->use_nested)
        put("%base = OpFunctionParameter %u64\n");
    for (uint32_t i = 0; i < call->n_in; ++i)
        if (call->in_active[i])
            fmt("%a$u = OpFunctionParameter %$t\n", i,
                jitc_var(call->outer_in[i]));
    fmt("%entry = OpLabel\n"
        "; Call: $s\n", call->name.c_str());

    // Local arrays of the callable are declared in its first block
    size_t locals_target = jitc_vulkan_locals_target;
    jitc_vulkan_locals_target = buffer.size();

    jitc_call_bind_slots(call, inst);

    for (size_t i = 0; i < schedule.size(); ++i) {
        ScheduledVariable &sv = schedule[i];
        Variable *v = jitc_var(sv.index);
        VarType vt = (VarType) v->type;
        VarKind kind = (VarKind) v->kind;

        jitc_vulkan_declare_type(vt);

        if (kind == VarKind::Counter) {
            fmt("%$v = OpCopyObject %$t %index\n", v, v);
        } else if (kind == VarKind::CallInput) {
            uint32_t in_i = 0;
            while (call->inner_in[in_i] != sv.index)
                in_i++;
            fmt("%$v = OpCopyObject %$t %a$u\n", v, v, in_i);
        } else if (kind == VarKind::CallSelf) {
            fmt("%$v = OpCopyObject %u32 %self\n", v);
        } else if (v->is_evaluated() || (vt == VarType::Pointer && kind == VarKind::Literal)) {
            // Captured value, read from the call data of this instance
            uint32_t offset = jitc_call_slot_rel_offset(call, inst, v, sv.index);
            VarType lt = vt == VarType::Bool ? VarType::UInt8 : vt;
            jitc_vulkan_declare_ptr_type(lt);
            jitc_vulkan_declare_const(VarType::UInt64, offset);
            fmt("%$v_ca = OpIAdd %u64 %data %c_u64_$u\n"
                "%$v_cp = OpConvertUToPtr %ptr_$s %$v_ca\n",
                v, offset,
                v, type_name_spirv[(int) lt], v);
            jitc_vulkan_render_load(v, "c", "");
        } else {
            jitc_vulkan_render(v);
        }
    }

    if (ret == "void") {
        put("OpReturn\n");
    } else {
        fmt("%result = OpCompositeConstruct %$s", ret.c_str());
        for (uint32_t i = 0; i < call->n_out; ++i)
            if (call->out_offset[i] != (uint32_t) -1)
                fmt(" %$v", jitc_var(call->inner_out[inst * call->n_out + i]));
        put("\nOpReturnValue %result\n");
    }
    put("OpFunctionEnd\n");

    jitc_vulkan_locals_target = locals_target;
}

/// Append the 32 hex digits of a callable's body hash, which name it
static void jitc_vulkan_put_hash(XXH128_hash_t hash) {
    char tmp[33];
    snprintf(tmp, sizeof(tmp), "%016llx%016llx",
             (unsigned long long) hash.high64,
             (unsigned long long) hash.low64);
    put(tmp, 32);
}

/// Name of the base pointer holding the kernel's combined call data
static void jitc_vulkan_call_base(char *base, size_t size) {
    if (callable_depth == 0)
        snprintf(base, size, "r%u", call_buffer.base_reg);
    else
        snprintf(base, size, "base");
}

void jitc_var_call_getter_assemble_vulkan(Variable *v, const Variable *index,
                                          const Variable *mask) {
    GetterData *gd = (GetterData *) v->data;
    VarType vt = (VarType) v->type,
            lt = vt == VarType::Bool ? VarType::UInt8 : vt;
    uint32_t tsize = type_size[(int) lt];
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    char base[16];
    jitc_vulkan_call_base(base, sizeof(base));

    jitc_vulkan_declare_ptr_type(lt);
    jitc_vulkan_declare_const(VarType::UInt64, tsize);
    jitc_vulkan_declare_const(VarType::UInt64, gd->header_offset);
    fmt("%$v_gi = OpUConvert %u64 %$v\n"
        "%$v_go = OpIMul %u64 %$v_gi %c_u64_$u\n"
        "%$v_gb = OpIAdd %u64 %$s %c_u64_$u\n"
        "%$v_ga = OpIAdd %u64 %$v_gb %$v_go\n"
        "%$v_gp = OpConvertUToPtr %ptr_$s %$v_ga\n",
        v, index,
        v, v, tsize,
        v, base, gd->header_offset,
        v, v, v,
        v, type_name_spirv[(int) lt], v);

    if (is_unmasked) {
        jitc_vulkan_render_load(v, "g", "");
    } else {
        jitc_vulkan_declare_const(vt, 0);
        jitc_vulkan_render_if(v, mask);
        jitc_vulkan_render_load(v, "g", "_l");
        jitc_vulkan_render_endif(v);
        fmt("%$v = OpPhi %$t %$v_l %$v_d %c_$t_0 %$v_f\n", v, v, v, v, v, v);
    }
}

void jitc_var_call_assemble_vulkan(CallData *call, uint32_t /*call_reg*/,
                                   uint32_t self_reg, uint32_t /*mask_reg*/) {
    Variable *v = jitc_var(call->id),
             *mask = jitc_var(v->dep[1]);
    bool is_masked = !mask->is_literal() || mask->literal != 1,
         has_slots = !call->slots.empty();

    std::string ret, fn;
    jitc_vulkan_declare_call_types(call, ret, fn);

    fmt("; Call: $s\n", call->name.c_str());
    if (is_masked)
        jitc_vulkan_render_if(v, mask);

    char base[16];
    jitc_vulkan_call_base(base, sizeof(base));

    // Load the offset table entry '%<v>_oe' of 'self'. Its low 32 bits hold
    // the callable index '%<v>_ci', and its high 32 bits the offset of the
    // instance's data '%<v>_cd' in the call-data buffer.
    if (call->n_inst != 1 || has_slots) {
        jitc_vulkan_declare_ptr_type(VarType::UInt64);
        jitc_vulkan_declare_const(VarType::UInt64, 8);
        jitc_vulkan_declare_const(VarType::UInt64, call->offset_base);
        fmt("%$v_oi = OpUConvert %u64 %r$u\n"
            "%$v_oo = OpIMul %u64 %$v_oi %c_u64_8\n"
            "%$v_ob = OpIAdd %u64 %$s %c_u64_$u\n"
            "%$v_oa = OpIAdd %u64 %$v_ob %$v_oo\n"
            "%$v_op = OpConvertUToPtr %ptr_u64 %$v_oa\n"
            "%$v_oe = OpLoad %u64 %$v_op Aligned 8\n",
            v, self_reg,
            v, v,
            v, base, call->offset_base,
            v, v, v,
            v, v,
            v, v);
        if (call->n_inst != 1)
            fmt("%$v_ci = OpUConvert %u32 %$v_oe\n", v, v);
    }

    if (has_slots) {
        jitc_vulkan_declare_const(VarType::UInt32, 32);
        fmt("%$v_oh = OpShiftRightLogical %u64 %$v_oe %c_u32_32\n"
            "%$v_cd = OpIAdd %u64 %$s %$v_oh\n",
            v, v, v, base, v);
    } else {
        jitc_vulkan_declare_const(VarType::UInt64, 0);
    }

    auto put_call = [&](XXH128_hash_t hash) {
        fmt("OpFunctionCall %$s %func_", ret.c_str());
        jitc_vulkan_put_hash(hash);
        fmt(" %$s", callable_depth == 0 ? "r0" : "index");
        fmt(" %r$u", self_reg);
        if (has_slots)
            fmt(" %$v_cd", v);
        else
            put(" %c_u64_0");
        if (call->use_nested)
            fmt(" %$s", base);
        for (uint32_t i = 0; i < call->n_in; ++i)
            if (call->in_active[i])
                fmt(" %$v", jitc_var(call->outer_in[i]));
        put('\n');
    };

    if (call->n_inst == 1) {
        fmt("%$v_rv = ", v);
        put_call(call->inst_hash[0]);
    } else {
        // One case per distinct callable
        std::vector<XXH128_hash_t> targets;
        for (const XXH128_hash_t &h : call->inst_hash) {
            bool found = false;
            for (const XXH128_hash_t &h2 : targets)
                found |= h.high64 == h2.high64 && h.low64 == h2.low64;
            if (!found)
                targets.push_back(h);
        }

        fmt("OpSelectionMerge %$v_m None\n"
            "OpSwitch %$v_ci %$v_z", v, v, v);
        for (size_t k = 0; k < targets.size(); ++k) {
            put(" @");
            jitc_vulkan_put_hash(targets[k]);
            fmt(" %$v_c$u", v, (uint32_t) k);
        }
        put('\n');

        for (size_t k = 0; k < targets.size(); ++k) {
            fmt("%$v_c$u = OpLabel\n"
                "%$v_r$u = ", v, (uint32_t) k, v, (uint32_t) k);
            put_call(targets[k]);
            fmt("OpBranch %$v_m\n", v);
        }

        // Unreachable: the call mask excludes invalid instances
        fmt("%$v_z = OpLabel\n"
            "OpBranch %$v_m\n"
            "%$v_m = OpLabel\n", v, v, v);

        if (ret != "void") {
            fmt("%$v_rv = OpPhi %$s", v, ret.c_str());
            for (size_t k = 0; k < targets.size(); ++k)
                fmt(" %$v_r$u %$v_c$u", v, (uint32_t) k, v, (uint32_t) k);
            fmt(" %null_$s %$v_z\n", ret.c_str(), v);
        }
    }

    // Extract the live outputs. Masked lanes produce zero.
    for (uint32_t i = 0, k = 0; i < call->n_out; ++i) {
        if (call->out_offset[i] == (uint32_t) -1)
            continue;
        Variable *vo = jitc_var(call->outer_out[i]);
        if (vo && vo->reg_index && vo->param_type != ParamType::Input) {
            jitc_vulkan_declare_type((VarType) vo->type);
            fmt("%$v$s = OpCompositeExtract %$t %$v_rv $u\n", vo,
                is_masked ? "_cl" : "", vo, v, k);
        }
        k++;
    }

    if (is_masked) {
        jitc_vulkan_render_endif(v);
        for (uint32_t i = 0; i < call->n_out; ++i) {
            if (call->out_offset[i] == (uint32_t) -1)
                continue;
            Variable *vo = jitc_var(call->outer_out[i]);
            if (!vo || !vo->reg_index || vo->param_type == ParamType::Input)
                continue;
            jitc_vulkan_declare_const((VarType) vo->type, 0);
            fmt("%$v = OpPhi %$t %$v_cl %$v_d %c_$t_0 %$v_f\n", vo, vo, vo,
                v, vo, v);
        }
    }
}
