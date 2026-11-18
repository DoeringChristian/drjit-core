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

// Forward declaration
static void jitc_vulkan_render(Variable *v);

/// Declare the SPIR-V type of 'vt'
static void jitc_vulkan_declare_type(VarType vt) {
    switch (vt) {
        case VarType::Void:    fmt_intrinsic("%void = OpTypeVoid"); break;
        case VarType::Bool:    fmt_intrinsic("%bool = OpTypeBool"); break;
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

/// Declare the 64-bit constant '%c_u64_<value>'
static void jitc_vulkan_declare_u64(uint32_t value) {
    fmt_intrinsic("%c_u64_$u = OpConstant %u64 $u", value, value);
}

/// Load the entry of ``v`` in the kernel parameter array into '%<v>_d'
static void jitc_vulkan_render_param(const Variable *v) {
    jitc_vulkan_declare_u64(v->param_offset);
    fmt("%$v_s = OpIAdd %u64 %params %c_u64_$u\n"
        "%$v_sp = OpConvertUToPtr %ptr_u64 %$v_s\n"
        "%$v_d = OpLoad %u64 %$v_sp Aligned 8\n",
        v, v->param_offset, v, v, v, v);
}

/// Compute a pointer '%<v>_p' to the element of ``v`` processed by this thread
static void jitc_vulkan_render_element_ptr(const Variable *v) {
    VarType vt = (VarType) v->type;
    if (v->is_array() || vt == VarType::Bool)
        jitc_raise("jitc_vulkan_render_element_ptr(): memory accesses to "
                   "variable r%u (type=%s) are not yet supported by the "
                   "Vulkan backend.", v->reg_index, type_name[v->type]);

    jitc_vulkan_declare_ptr_type(vt);

    if (v->size > 1) {
        uint32_t isize = type_size[v->type];
        jitc_vulkan_declare_u64(isize);
        fmt("%$v_i = OpUConvert %u64 %r0\n"
            "%$v_o = OpIMul %u64 %$v_i %c_u64_$u\n"
            "%$v_e = OpIAdd %u64 %$v_d %$v_o\n"
            "%$v_p = OpConvertUToPtr %ptr_$t %$v_e\n",
            v, v, v, isize, v, v, v, v, v, v);
    } else {
        fmt("%$v_p = OpConvertUToPtr %ptr_$t %$v_d\n", v, v, v);
    }
}

void jitc_vulkan_render_prologue(ThreadState *ts) {
    const VulkanDevice &dev = state.vulkan_devices[ts->device];

    put("OpCapability Shader\n"
        "OpCapability Int64\n"
        "OpCapability PhysicalStorageBufferAddresses\n"
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
    fmt_intrinsic("%c_u32_0 = OpConstant %u32 0");
    const VulkanDevice &dev = state.vulkan_devices[ts->device];
    fmt_intrinsic("%c_u32_$u = OpConstant %u32 $u", dev.block_size,
                  dev.block_size);

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

        jitc_vulkan_declare_type((VarType) v->type);

        if (ptype == ParamType::Input) {
            if (v->is_literal()) {
                // Pointer literals must be loaded from the parameter array
                // (not inlined) so frozen function replay can update them
                if ((VarType) v->type == VarType::Pointer) {
                    jitc_vulkan_render_param(v);
                    fmt("%$v = OpCopyObject %u64 %$v_d\n", v, v);
                } else {
                    jitc_vulkan_render(v);
                }
                continue;
            }

            jitc_vulkan_render_param(v);
            jitc_vulkan_render_element_ptr(v);
            fmt("%$v = OpLoad %$t %$v_p Aligned $u\n", v, v, v,
                (uint32_t) type_size[v->type]);
            continue;
        }

        jitc_vulkan_render(v);

        if (ptype == ParamType::Output) {
            jitc_vulkan_render_param(v);
            jitc_vulkan_render_element_ptr(v);
            fmt("OpStore %$v_p %$v Aligned $u\n", v, v,
                (uint32_t) type_size[v->type]);
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

static void jitc_vulkan_render(Variable *v) {
    switch ((VarKind) v->kind) {
        case VarKind::Nop:
            break;

        case VarKind::Literal:
            // Constants are named after their value, so that the body of a
            // callable (which determines its hash) references the value, and
            // the declarations of different units coincide
            if ((VarType) v->type == VarType::Bool) {
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

        case VarKind::Add: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            fmt("%$v = $s %$t %$v %$v\n", v,
                jitc_is_float(v) ? "OpFAdd" : "OpIAdd", v, a0, a1);
            break;
        }

        default:
            jitc_raise("jitc_vulkan_render(): unhandled VarKind=%u (%s) for "
                       "variable r%u (type=%s).",
                       (uint32_t) v->kind, var_kind_name[v->kind],
                       v->reg_index, type_name[v->type]);
    }
}
