/*
    src/vulkan_packet.cpp -- Specialized memory operations that read or write
    multiple adjacent values at once (SPIR-V codegen).

    A packet is transferred in naturally aligned chunks of up to 16 bytes,
    which are reinterpreted as vectors of elements (see metal_packet.cpp).

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "vulkan_packet.h"
#include "eval.h"
#include "internal.h"
#include "var.h"
#include "op.h"
#include "log.h"
#include "strbuf.h"
#include "vulkan_eval.h"
#include <cstring>

/// Declare and return the SPIR-V type of a chunk of ``bytes`` bytes
static const char *jitc_vulkan_packet_word_type(uint32_t bytes) {
    const char *name;
    switch (bytes) {
        case 16:
            jitc_vulkan_declare_type(VarType::UInt32);
            fmt_intrinsic("%v4u32 = OpTypeVector %u32 4");
            name = "v4u32";
            break;
        case 8:
            jitc_vulkan_declare_type(VarType::UInt32);
            fmt_intrinsic("%v2u32 = OpTypeVector %u32 2");
            name = "v2u32";
            break;
        case 4:
            jitc_vulkan_declare_type(VarType::UInt32);
            name = "u32";
            break;
        case 2:
            jitc_vulkan_declare_type(VarType::UInt16);
            name = "u16";
            break;
        default:
            jitc_fail("jitc_vulkan_packet_word_type(): unsupported chunk "
                      "width %u!", bytes);
    }
    fmt_intrinsic("%ptr_$s = OpTypePointer PhysicalStorageBuffer %$s", name,
                  name);
    return name;
}

/// Widest naturally aligned chunk size (in bytes) for transferring a packet of
/// ``total_bytes`` bytes with element size ``tsize``
static uint32_t jitc_vulkan_packet_chunk_bytes(uint32_t tsize,
                                               uint32_t total_bytes) {
    uint32_t bytes = 4 * tsize > 16 ? 16 : 4 * tsize;
    while (total_bytes % bytes)
        bytes /= 2;
    return bytes;
}

/// Layout of a packet of ``count`` elements of type ``vt``
struct VulkanPacket {
    VarType et;          // Element type in memory (booleans are stored as bytes)
    uint32_t tsize;      // Element size
    uint32_t chunk;      // Bytes per load/store
    uint32_t lanes;      // Elements per chunk
    uint32_t n_chunks;   // Number of loads/stores
    const char *word;    // SPIR-V type of a chunk
    char vec[16];        // SPIR-V type of the elements of a chunk
    const char *cast;    // Opcode converting between 'word' and 'vec'

    VulkanPacket(VarType vt, uint32_t count) {
        et = vt == VarType::Bool ? VarType::UInt8 : vt;
        tsize = type_size[(int) et];
        chunk = jitc_vulkan_packet_chunk_bytes(tsize, count * tsize);
        lanes = chunk / tsize;
        n_chunks = count * tsize / chunk;
        word = jitc_vulkan_packet_word_type(chunk);

        // Packets have an even number of elements, hence 'lanes' is >= 2
        const char *elem = type_name_spirv[(int) et];
        jitc_vulkan_declare_type(et);
        snprintf(vec, sizeof(vec), "v%u%s", lanes, elem);
        fmt_intrinsic("%$s = OpTypeVector %$s $u", vec, elem, lanes);
        cast = strcmp(word, vec) == 0 ? "OpCopyObject" : "OpBitcast";
    }
};

/// Compute the address '%<v>_ka' of the packet processed by this thread, which
/// starts at element 'index * scale' of the array at 'ptr'
static void jitc_vulkan_render_packet_base(const Variable *v,
                                           const Variable *ptr,
                                           const Variable *index,
                                           uint32_t scale) {
    jitc_vulkan_declare_const(VarType::UInt64, scale);
    fmt("%$v_ki = OpUConvert %u64 %$v\n"
        "%$v_ko = OpIMul %u64 %$v_ki %c_u64_$u\n"
        "%$v_ka = OpIAdd %u64 %$v %$v_ko\n",
        v, index,
        v, v, scale,
        v, ptr, v);
}

/// Compute the pointer '%<v>_kp<c>' to chunk ``c`` of the packet
static void jitc_vulkan_render_packet_ptr(const Variable *v,
                                          const VulkanPacket &p, uint32_t c) {
    jitc_vulkan_declare_const(VarType::UInt64, c * p.chunk);
    fmt("%$v_kb$u = OpIAdd %u64 %$v_ka %c_u64_$u\n"
        "%$v_kp$u = OpConvertUToPtr %ptr_$s %$v_kb$u\n",
        v, c, v, c * p.chunk,
        v, c, p.word, v, c);
}

void jitc_vulkan_render_gather_packet(const Variable *v, const Variable *ptr,
                                      const Variable *index,
                                      const Variable *mask) {
    bool is_masked = !mask->is_literal() || mask->literal != 1;
    uint32_t count = (uint32_t) v->literal;
    VulkanPacket p((VarType) v->type, count);

    jitc_vulkan_render_packet_base(v, ptr, index, count * p.tsize);

    // Load the chunks into '%<v>_kw<c>', zero for masked lanes
    if (is_masked) {
        fmt_intrinsic("%null_$s = OpConstantNull %$s", p.word, p.word);
        jitc_vulkan_render_if(v, mask);
    }

    for (uint32_t c = 0; c < p.n_chunks; ++c) {
        jitc_vulkan_render_packet_ptr(v, p, c);
        fmt("%$v_$s$u = OpLoad %$s %$v_kp$u Aligned $u\n", v,
            is_masked ? "kl" : "kw", c, p.word, v, c, p.chunk);
    }

    if (is_masked) {
        jitc_vulkan_render_endif(v);
        for (uint32_t c = 0; c < p.n_chunks; ++c)
            fmt("%$v_kw$u = OpPhi %$s %$v_kl$u %$v_d %null_$s %$v_f\n",
                v, c, p.word, v, c, v, p.word, v);
    }

    // Reinterpret the chunks and extract the elements into '%<v>_out_<i>'
    for (uint32_t c = 0; c < p.n_chunks; ++c)
        fmt("%$v_kx$u = $s %$s %$v_kw$u\n", v, c, p.cast, p.vec, v, c);

    bool is_bool = jitc_is_bool(v);
    if (is_bool)
        jitc_vulkan_declare_const(VarType::UInt8, 0);

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t c = i / p.lanes,
                 l = i % p.lanes;
        if (is_bool)
            fmt("%$v_ke$u = OpCompositeExtract %u8 %$v_kx$u $u\n"
                "%$v_out_$u = OpINotEqual %bool %$v_ke$u %c_u8_0\n",
                v, i, v, c, l,
                v, i, v, i);
        else
            fmt("%$v_out_$u = OpCompositeExtract %$t %$v_kx$u $u\n",
                v, i, v, v, c, l);
    }
}

void jitc_vulkan_render_scatter_packet(const Variable *v, const Variable *ptr,
                                       const Variable *index,
                                       const Variable *mask) {
    PacketScatterData *psd              = (PacketScatterData *) v->data;
    const std::vector<uint32_t> &values = psd->values;
    const Variable *v0                  = jitc_var(values[0]);

    // op.cpp splits packet scatter-reductions into scalar ones on Vulkan
    if (psd->op != ReduceOp::Identity)
        jitc_fail("jitc_vulkan_render_scatter_packet(): packet "
                  "scatter-reductions are not supported by the Vulkan "
                  "backend.");

    bool is_masked = !mask->is_literal() || mask->literal != 1,
         is_bool = jitc_is_bool(v0);
    uint32_t count = (uint32_t) values.size();
    VulkanPacket p((VarType) v0->type, count);

    // The scatter index is already scaled by 'count' (see op.cpp)
    jitc_vulkan_render_packet_base(v, ptr, index, p.tsize);

    // Assemble the chunks '%<v>_kw<c>'
    if (is_bool) {
        jitc_vulkan_declare_const(VarType::UInt8, 0);
        jitc_vulkan_declare_const(VarType::UInt8, 1);
        for (uint32_t i = 0; i < count; ++i)
            fmt("%$v_kv$u = OpSelect %u8 %$v %c_u8_1 %c_u8_0\n", v, i,
                jitc_var(values[i]));
    }

    for (uint32_t c = 0; c < p.n_chunks; ++c) {
        fmt("%$v_kx$u = OpCompositeConstruct %$s", v, c, p.vec);
        for (uint32_t l = 0; l < p.lanes; ++l) {
            uint32_t i = c * p.lanes + l;
            if (is_bool)
                fmt(" %$v_kv$u", v, i);
            else
                fmt(" %$v", jitc_var(values[i]));
        }
        fmt("\n%$v_kw$u = $s %$s %$v_kx$u\n", v, c, p.cast, p.word, v, c);
    }

    if (is_masked)
        jitc_vulkan_render_if(v, mask);

    for (uint32_t c = 0; c < p.n_chunks; ++c) {
        jitc_vulkan_render_packet_ptr(v, p, c);
        fmt("OpStore %$v_kp$u %$v_kw$u Aligned $u\n", v, c, v, c, p.chunk);
    }

    if (is_masked)
        jitc_vulkan_render_endif(v);
}
