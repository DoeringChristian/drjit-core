/*
    src/vulkan_scatter.h -- Scatter / scatter-reduce SPIR-V codegen.

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

struct Variable;

/// Emit SPIR-V code for a scatter or scatter-reduce operation.
extern void jitc_vulkan_render_scatter(Variable *v);

/// Emit SPIR-V code for an atomic increment (ScatterInc).
extern void jitc_vulkan_render_scatter_inc(Variable *v);

/// Emit SPIR-V code for an atomic exchange (ScatterExch).
extern void jitc_vulkan_render_scatter_exch(Variable *v);

/// Emit SPIR-V code for an atomic compare-and-swap (ScatterCAS).
extern void jitc_vulkan_render_scatter_cas(Variable *v);
