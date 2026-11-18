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
#include "vulkan.h"

// Forward declaration
static void jitc_vulkan_render(Variable *v);

/// Declare the SPIR-V type of 'vt'
static void jitc_vulkan_declare_type(VarType vt) {
    switch (vt) {
        case VarType::Void:    fmt_intrinsic("%void = OpTypeVoid"); break;
        case VarType::Bool:    fmt_intrinsic("%bool = OpTypeBool"); break;
        case VarType::UInt8:   fmt_intrinsic("%u8 = OpTypeInt 8 0"); break;
        case VarType::Int32:   fmt_intrinsic("%i32 = OpTypeInt 32 1"); break;
        case VarType::UInt32:  fmt_intrinsic("%u32 = OpTypeInt 32 0"); break;
        case VarType::Int64:   fmt_intrinsic("%i64 = OpTypeInt 64 1"); break;
        case VarType::UInt64:
        case VarType::Pointer: fmt_intrinsic("%u64 = OpTypeInt 64 0"); break;
        case VarType::Float32: fmt_intrinsic("%f32 = OpTypeFloat 32"); break;
        default:
            jitc_raise("jitc_vulkan_declare_type(): type \"%s\" is not yet "
                       "supported by the Vulkan backend.", type_name[(int) vt]);
    }
}

/// Declare the type of pointers to 'vt' in device memory
static void jitc_vulkan_declare_ptr_type(VarType vt) {
    jitc_vulkan_declare_type(vt);
    const char *name = type_name_spirv[(int) vt];
    fmt_intrinsic("%ptr_$s = OpTypePointer PhysicalStorageBuffer %$s",
                  name, name);
}

/// Declare the constant '%c_<type>_<value>' of type 'vt'
static void jitc_vulkan_declare_const(VarType vt, uint32_t value) {
    jitc_vulkan_declare_type(vt);
    const char *name = type_name_spirv[(int) vt];
    fmt_intrinsic("%c_$s_$u = OpConstant %$s $u", name, value, name, value);
}

/// Unsigned integer type with the same size as 'vt'
static VarType jitc_vulkan_bin_type(VarType vt) {
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

/// Compute a pointer '%<v>_ep' to the element of ``v`` processed by this
/// thread. Booleans are stored as bytes.
static void jitc_vulkan_render_element_ptr(const Variable *v) {
    if (v->is_array())
        jitc_raise("jitc_vulkan_render_element_ptr(): memory accesses to "
                   "variable r%u (type=%s) are not yet supported by the "
                   "Vulkan backend.", v->reg_index, type_name[v->type]);

    VarType vt = (VarType) v->type;
    if (vt == VarType::Bool)
        vt = VarType::UInt8;
    jitc_vulkan_declare_ptr_type(vt);
    const char *name = type_name_spirv[(int) vt];

    if (v->size > 1) {
        uint32_t isize = type_size[v->type];
        jitc_vulkan_declare_const(VarType::UInt64, isize);
        fmt("%$v_ei = OpUConvert %u64 %r0\n"
            "%$v_eo = OpIMul %u64 %$v_ei %c_u64_$u\n"
            "%$v_ea = OpIAdd %u64 %$v_pd %$v_eo\n"
            "%$v_ep = OpConvertUToPtr %ptr_$s %$v_ea\n",
            v, v, v, isize, v, v, v, v, name, v);
    } else {
        fmt("%$v_ep = OpConvertUToPtr %ptr_$s %$v_pd\n", v, name, v);
    }
}

void jitc_vulkan_render_prologue(ThreadState *ts) {
    const VulkanDevice &dev = state.vulkan_devices[ts->device];

    put("OpCapability Shader\n"
        "OpCapability Int64\n"
        "OpCapability PhysicalStorageBufferAddresses\n"
        "OpCapability StorageBuffer8BitAccess\n"
        "%glsl = OpExtInstImport \"GLSL.std.450\"\n"
        "OpMemoryModel PhysicalStorageBuffer64 GLSL450\n"
        "OpEntryPoint GLCompute %main "
        "\"drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\" %gid %nwg %push\n");
    fmt("OpExecutionMode %main LocalSize $u 1 1\n", dev.block_size);
    put("OpDecorate %gid BuiltIn GlobalInvocationId\n"
        "OpDecorate %nwg BuiltIn NumWorkgroups\n"
        "OpDecorate %Push Block\n"
        "OpMemberDecorate %Push 0 Offset 0\n");
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

    // The number of workgroups is limited by the device, hence threads
    // process the elements in a grid-stride loop
    fmt("%main = OpFunction %void None %fn_void\n"
        "%entry = OpLabel\n"
        "%gid_v = OpLoad %v3u32 %gid\n"
        "%r0_start = OpCompositeExtract %u32 %gid_v 0\n"
        "%nwg_v = OpLoad %v3u32 %nwg\n"
        "%nwg_x = OpCompositeExtract %u32 %nwg_v 0\n"
        "%stride = OpIMul %u32 %nwg_x %c_u32_$u\n"
        "%params_p = OpAccessChain %ptr_push_u64 %push %c_u32_0\n"
        "%params = OpLoad %u64 %params_p\n"
        "%size_p = OpConvertUToPtr %ptr_u32 %params\n"
        "%size = OpLoad %u32 %size_p Aligned 8\n"
        "OpBranch %head\n"
        "%head = OpLabel\n"
        "%r0 = OpPhi %u32 %r0_start %entry %r0_next %next\n"
        "%active = OpULessThan %bool %r0 %size\n"
        "OpLoopMerge %done %next None\n"
        "OpBranchConditional %active %body %done\n"
        "%body = OpLabel\n", dev.block_size);

    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        Variable *v = jitc_var(index);
        ParamType ptype = (ParamType) v->param_type;
        bool is_bool = (VarType) v->type == VarType::Bool;

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
            jitc_vulkan_render_element_ptr(v);
            if (is_bool) {
                // 8-bit storage only permits loads, stores, and conversions
                fmt("%$v_bm = OpLoad %u8 %$v_ep Aligned 1\n"
                    "%$v_bw = OpUConvert %u32 %$v_bm\n"
                    "%$v = OpINotEqual %bool %$v_bw %c_u32_0\n",
                    v, v, v, v, v, v);
            } else {
                fmt("%$v = OpLoad %$t %$v_ep Aligned $u\n", v, v, v,
                    (uint32_t) type_size[v->type]);
            }
            continue;
        }

        jitc_vulkan_render(v);

        if (ptype == ParamType::Output) {
            jitc_vulkan_render_param(v);
            jitc_vulkan_render_element_ptr(v);
            if (is_bool) {
                jitc_vulkan_declare_const(VarType::UInt32, 1);
                fmt("%$v_bw = OpSelect %u32 %$v %c_u32_1 %c_u32_0\n"
                    "%$v_bm = OpUConvert %u8 %$v_bw\n"
                    "OpStore %$v_ep %$v_bm Aligned 1\n",
                    v, v, v, v, v, v);
            } else {
                fmt("OpStore %$v_ep %$v Aligned $u\n", v, v,
                    (uint32_t) type_size[v->type]);
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

static void jitc_vulkan_render_unhandled(const Variable *v) {
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

static void jitc_vulkan_render(Variable *v) {
    Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
             *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
             *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr;
    VarType vt = (VarType) v->type;

    switch ((VarKind) v->kind) {
        case VarKind::Nop:
            break;

        case VarKind::Literal:
            // Constants are named after their value, so that the body of a
            // callable (which determines its hash) references the value, and
            // the declarations of different units coincide
            if (vt == VarType::Bool) {
                fmt_intrinsic("%c_bool_$u = $s %bool", v->literal ? 1 : 0,
                              v->literal ? "OpConstantTrue" : "OpConstantFalse");
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

        // -- Arithmetic --
        case VarKind::Add: jitc_vulkan_render_binary(v, "OpFAdd", "OpIAdd", "OpIAdd"); break;
        case VarKind::Sub: jitc_vulkan_render_binary(v, "OpFSub", "OpISub", "OpISub"); break;
        case VarKind::Mul: jitc_vulkan_render_binary(v, "OpFMul", "OpIMul", "OpIMul"); break;
        case VarKind::Div:
        case VarKind::DivApprox:
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
            if (jitc_is_single(v)) {
                jitc_vulkan_declare_const(VarType::UInt32, 0x7fc00000u);
                fmt("%$v_m = OpExtInst %$t %glsl $s %$v %$v\n"
                    "%$v_a = OpIsNan %bool %$v\n"
                    "%$v_b = OpIsNan %bool %$v\n"
                    "%$v_o = OpLogicalOr %bool %$v_a %$v_b\n"
                    "%$v_n = OpBitcast %$t %c_u32_$u\n"
                    "%$v = OpSelect %$t %$v_o %$v_n %$v_m\n",
                    v, v, is_min ? "FMin" : "FMax", a0, a1,
                    v, a0,
                    v, a1,
                    v, v, v,
                    v, v, 0x7fc00000u,
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
            jitc_vulkan_render_ext(v, "Sqrt", 1);
            break;

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

        default:
            jitc_vulkan_render_unhandled(v);
    }
}
