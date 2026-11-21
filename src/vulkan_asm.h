/*
    src/vulkan_asm.h -- Assembler for the SPIR-V text of the Vulkan backend

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/**
 * \brief Assemble SPIR-V text into a SPIR-V 1.5 module
 *
 * Accepts the text format of SPIRV-Tools ("spirv-as"), with one instruction
 * per line, and numbers IDs in the order of their first appearance. On
 * failure, the function returns \c false and stores a description of the
 * problem in \c error and the (zero-based) line in \c line.
 */
extern bool vulkan_asm(const char *source, size_t size,
                       std::vector<uint8_t> &out, std::string &error,
                       size_t &line);
