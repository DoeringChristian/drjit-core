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
