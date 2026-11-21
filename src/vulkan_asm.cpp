/*
    src/vulkan_asm.cpp -- Assembler for the SPIR-V text of the Vulkan backend

    The instruction and operand tables are generated from the SPIR-V grammar
    (resources/spirv_grammar.py)

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "vulkan_asm.h"
#include "resources/spirv_grammar.h"
#include <drjit-core/half.h>
#include <tsl/robin_map.h>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace {

struct AsmError { std::string message; };

struct Token {
    const char *s;
    size_t n;
    std::string_view view() const { return { s, n }; }
};

/// Information about an ID that is needed to encode literals
struct IdInfo {
    uint32_t type = 0;   // Type of a value
    uint32_t width = 0;  // Bit width of a scalar numeric type
    bool is_float = false, is_signed = false;
    int ext_set = -1;    // Extended instruction set of an OpExtInstImport
};

// Opcodes whose results the assembler keeps track of
const uint32_t OpExtInstImport = 11, OpTypeInt = 21, OpTypeFloat = 22;

/// Binary search for 'name' in a range sorted by name
template <typename T>
const T *lookup(const T *begin, const T *end, std::string_view name) {
    while (begin < end) {
        const T *mid = begin + (end - begin) / 2;
        int c = name.compare(mid->name);
        if (c == 0)
            return mid;
        else if (c < 0)
            end = mid;
        else
            begin = mid + 1;
    }
    return nullptr;
}

[[noreturn]] void error(const char *message, const Token *t = nullptr) {
    std::string s = message;
    if (t)
        s += " \"" + std::string(t->view()) + "\"";
    throw AsmError{ std::move(s) };
}

class Assembler {
public:
    Assembler() : words(5, 0), info(1) { }

    void line(const char *s, const char *e) {
        tokenize(s, e);
        if (tokens.empty())
            return;

        size_t i = 0;
        result = nullptr;
        if (tokens.size() >= 2 && tokens[1].view() == "=") {
            result = &tokens[0];
            i = 2;
        }
        if (i == tokens.size())
            error("expected an opcode");

        const SpirvInstruction *inst =
            lookup(spirv_instructions, std::end(spirv_instructions),
                   tokens[i].view());
        if (!inst)
            error("unknown opcode", &tokens[i]);

        pos = i + 1;
        result_type = result_id = first_id = last_id = 0;
        size_t start = words.size();
        words.push_back(0);
        operands(inst->operand_offset, inst->operand_count);

        if (result && !result_id)
            error("instruction has no result ID", &tokens[i]);
        if (pos != tokens.size())
            error("unexpected operand", &tokens[pos]);

        size_t count = words.size() - start;
        if (count > 0xFFFF)
            error("instruction is too long");
        words[start] = (uint32_t) (count << 16) | inst->opcode;

        if (result_type)
            info[result_id].type = result_type;

        switch (inst->opcode) {
            case OpTypeInt:
                info[result_id].width = words[start + 2];
                info[result_id].is_signed = words[start + 3] != 0;
                break;

            case OpTypeFloat:
                info[result_id].width = words[start + 2];
                info[result_id].is_float = true;
                break;

            case OpExtInstImport: {
                    std::string_view name = tokens[pos - 1].view();
                    name = name.substr(1, name.size() - 2);
                    for (const SpirvExtInstSet &set : spirv_ext_inst_sets)
                        if (name == set.name)
                            info[result_id].ext_set =
                                (int) (&set - spirv_ext_inst_sets);
                }
                break;
        }
    }

    std::vector<uint32_t> finish() {
        words[0] = 0x07230203; // Magic number
        words[1] = 0x00010500; // SPIR-V 1.5 (Vulkan 1.2)
        words[3] = (uint32_t) info.size(); // ID bound
        return std::move(words);
    }

private:
    void tokenize(const char *p, const char *e) {
        tokens.clear();
        while (true) {
            while (p < e && (*p == ' ' || *p == '\t' || *p == '\r'))
                ++p;
            if (p == e || *p == ';')
                break;
            const char *start = p;
            if (*p == '"') {
                for (++p; p < e && *p != '"'; ++p) {
                    if (*p == '\\' && p + 1 < e)
                        ++p;
                }
                if (p == e)
                    error("unterminated string");
                ++p;
            } else {
                while (p < e && *p != ' ' && *p != '\t' && *p != '\r')
                    ++p;
            }
            tokens.push_back({ start, (size_t) (p - start) });
        }
    }

    const Token &next(SpirvKind kind) {
        if (pos == tokens.size()) {
            std::string s = "expected operand of kind ";
            error((s + spirv_kinds[kind].name).c_str());
        }
        return tokens[pos++];
    }

    /// Look up or assign the ID of the name in token 't'
    uint32_t id(const Token &t) {
        if (t.n < 2 || t.s[0] != '%')
            error("expected an ID", &t);
        auto [it, inserted] =
            ids.try_emplace(std::string_view(t.s + 1, t.n - 1),
                            (uint32_t) info.size());
        if (inserted)
            info.emplace_back();
        return it->second;
    }

    void operands(uint32_t offset, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            const SpirvOperand &o = spirv_operands[offset + i];
            if (o.kind == SpirvKindIdResult) {
                if (!result)
                    error("expected a result ID");
                result_id = id(*result);
                words.push_back(result_id);
            } else if (o.quantifier == SpirvOne) {
                operand(o.kind);
            } else if (o.quantifier == SpirvOptional) {
                if (pos < tokens.size())
                    operand(o.kind);
            } else {
                while (pos < tokens.size())
                    operand(o.kind);
            }
        }
    }

    void operand(SpirvKind kind) {
        const SpirvKindInfo &k = spirv_kinds[kind];

        switch (kind) {
            case SpirvKindIdResultType:
                result_type = id(next(kind));
                words.push_back(result_type);
                return;

            case SpirvKindLiteralString:
                literal_string(next(kind));
                return;

            case SpirvKindLiteralInteger:
                number(next(kind), 32, false, false, true);
                return;

            case SpirvKindLiteralFloat:
                number(next(kind), 32, true, false, false);
                return;

            case SpirvKindLiteralContextDependentNumber:
                numeric(next(kind), result_type);
                return;

            case SpirvKindLiteralExtInstInteger:
                ext_inst(next(kind));
                return;

            case SpirvKindLiteralSpecConstantOpInteger: {
                    const Token &t = next(kind);
                    std::string name = "Op" + std::string(t.view());
                    const SpirvInstruction *inst =
                        lookup(spirv_instructions,
                               std::end(spirv_instructions), name);
                    if (!inst)
                        error("unknown opcode", &t);
                    words.push_back(inst->opcode);
                    for (uint32_t i = 0; i < inst->operand_count; ++i) {
                        SpirvKind k2 =
                            spirv_operands[inst->operand_offset + i].kind;
                        if (k2 != SpirvKindIdResultType &&
                            k2 != SpirvKindIdResult)
                            operands(inst->operand_offset + i, 1);
                    }
                }
                return;

            case SpirvKindPairLiteralIntegerIdRef:
                // The literals of OpSwitch have the type of the selector
                numeric(next(kind), info[first_id].type);
                operand(SpirvKindIdRef);
                return;

            case SpirvKindPairIdRefLiteralInteger:
                operand(SpirvKindIdRef);
                operand(SpirvKindLiteralInteger);
                return;

            case SpirvKindPairIdRefIdRef:
                operand(SpirvKindIdRef);
                operand(SpirvKindIdRef);
                return;

            default:
                break;
        }

        const Token &t = next(kind);
        if (k.category == SpirvId) {
            uint32_t value = id(t);
            if (!first_id)
                first_id = value;
            last_id = value;
            words.push_back(value);
        } else if (k.category == SpirvValueEnum) {
            const SpirvEnumerant *e = enumerant(k, t.view());
            if (!e)
                error((std::string("invalid ") + k.name).c_str(), &t);
            words.push_back(e->value);
            operands(e->param_offset, e->param_count);
        } else if (k.category == SpirvBitEnum) {
            // Masks combine names with '|', the parameters of the set bits
            // follow in the order of the bits
            uint32_t value = 0;
            std::string_view s = t.view();
            while (true) {
                size_t sep = s.find('|');
                const SpirvEnumerant *e = enumerant(k, s.substr(0, sep));
                if (!e)
                    error((std::string("invalid ") + k.name).c_str(), &t);
                value |= e->value;
                if (sep == std::string_view::npos)
                    break;
                s = s.substr(sep + 1);
            }
            words.push_back(value);
            for (uint32_t bit = 1; bit && bit <= value; bit <<= 1) {
                if (!(value & bit))
                    continue;
                for (uint32_t i = k.enum_begin; i < k.enum_end; ++i) {
                    const SpirvEnumerant &e = spirv_enumerants[i];
                    if (e.value == bit) {
                        operands(e.param_offset, e.param_count);
                        break;
                    }
                }
            }
        } else {
            error("unsupported operand kind", &t);
        }
    }

    const SpirvEnumerant *enumerant(const SpirvKindInfo &k,
                                    std::string_view name) {
        return lookup(spirv_enumerants + k.enum_begin,
                    spirv_enumerants + k.enum_end, name);
    }

    void ext_inst(const Token &t) {
        int set = info[last_id].ext_set;
        if (set >= 0) {
            const SpirvExtInstSet &s = spirv_ext_inst_sets[set];
            const SpirvExtInst *inst =
                lookup(s.insts, s.insts + s.count, t.view());
            if (!inst)
                error("invalid extended instruction", &t);
            words.push_back(inst->opcode);
        } else {
            number(t, 32, false, false, false);
        }
    }

    void literal_string(const Token &t) {
        if (t.n < 2 || t.s[0] != '"')
            error("expected a string", &t);
        std::string s;
        for (size_t i = 1; i + 1 < t.n; ++i) {
            if (t.s[i] == '\\')
                ++i;
            s += t.s[i];
        }
        // Null-terminated, padded to a multiple of 4 bytes
        size_t start = words.size();
        words.resize(start + s.size() / 4 + 1, 0);
        for (size_t i = 0; i < s.size(); ++i)
            words[start + i / 4] |= (uint32_t) (uint8_t) s[i] << (8 * (i % 4));
    }

    /// Encode a literal of the numeric type with the given ID
    void numeric(const Token &t, uint32_t type) {
        const IdInfo &ti = info[type];
        if (!ti.width)
            error("literal requires a scalar numeric type", &t);
        number(t, ti.width, ti.is_float, ti.is_signed, false);
    }

    /**
     * Encode a literal number. Integers narrower than 32 bit are sign- or
     * zero-extended, 64-bit numbers take two words. 'any_sign' accepts
     * both negative and unsigned 32-bit values (LiteralInteger)
     */
    void number(const Token &t, uint32_t width, bool is_float, bool is_signed,
                bool any_sign) {
        if (is_float) {
            std::string s(t.view());
            char *end = nullptr;
            if (width == 16 || width == 32) {
                float f = std::strtof(s.c_str(), &end);
                uint32_t u;
                if (width == 16)
                    u = drjit::half(f).value;
                else
                    memcpy(&u, &f, sizeof(float));
                words.push_back(u);
            } else if (width == 64) {
                double d = std::strtod(s.c_str(), &end);
                uint64_t u;
                memcpy(&u, &d, sizeof(double));
                words.push_back((uint32_t) u);
                words.push_back((uint32_t) (u >> 32));
            } else {
                error("unsupported floating point width", &t);
            }
            if (s.empty() || end != s.c_str() + s.size())
                error("invalid floating point literal", &t);
            return;
        }

        const char *p = t.s, *e = t.s + t.n;
        bool neg = p < e && *p == '-';
        if (neg)
            ++p;
        bool hex = e - p > 2 && p[0] == '0' && (p[1] == 'x' || p[1] == 'X');
        if (hex)
            p += 2;
        if (p == e || width == 0 || width > 64)
            error("invalid integer literal", &t);

        uint64_t base = hex ? 16 : 10, v = 0;
        for (; p < e; ++p) {
            uint64_t d;
            if (*p >= '0' && *p <= '9')
                d = (uint64_t) (*p - '0');
            else if (hex && *p >= 'a' && *p <= 'f')
                d = (uint64_t) (*p - 'a' + 10);
            else if (hex && *p >= 'A' && *p <= 'F')
                d = (uint64_t) (*p - 'A' + 10);
            else
                error("invalid integer literal", &t);
            if (v > (UINT64_MAX - d) / base)
                error("integer literal is out of range", &t);
            v = v * base + d;
        }

        uint64_t mask = ~(uint64_t) 0 >> (64 - width),
                 min_neg = (uint64_t) 1 << (width - 1);
        bool valid;
        if (neg)
            valid = (is_signed || any_sign) && v <= min_neg;
        else if (hex || !is_signed || any_sign)
            valid = !(v & ~mask);
        else
            valid = v <= (mask >> 1);
        if (!valid)
            error("integer literal is out of range", &t);

        if (neg)
            v = (uint64_t) 0 - v;
        v &= mask;
        if (is_signed && width < 32 && (v >> (width - 1)) & 1)
            v |= ~mask;

        words.push_back((uint32_t) v);
        if (width > 32)
            words.push_back((uint32_t) (v >> 32));
    }

    std::vector<uint32_t> words;
    std::vector<IdInfo> info; // Indexed by ID, ID 0 is unused
    tsl::robin_map<std::string_view, uint32_t> ids;

    // State of the current line
    std::vector<Token> tokens;
    size_t pos = 0;
    const Token *result = nullptr;
    uint32_t result_type = 0, result_id = 0, first_id = 0, last_id = 0;
};

} // namespace

bool vulkan_asm(const char *source, size_t size, std::vector<uint8_t> &out,
                std::string &error, size_t &line) {
    Assembler as;
    const char *p = source, *end = source + size;
    line = 0;

    try {
        while (p < end) {
            const char *e = (const char *) memchr(p, '\n', (size_t) (end - p));
            if (!e)
                e = end;
            as.line(p, e);
            p = e + 1;
            line++;
        }
    } catch (const AsmError &e) {
        error = e.message;
        return false;
    }

    std::vector<uint32_t> words = as.finish();
    out.resize(words.size() * sizeof(uint32_t));
    memcpy(out.data(), words.data(), out.size());
    return true;
}
