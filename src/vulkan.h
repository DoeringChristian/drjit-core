/*
    src/vulkan.h -- Vulkan backend declarations

    Copyright (c) 2026 Christian Döring <christian.doering@tum.de>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"

#if defined(DRJIT_ENABLE_VULKAN)

/// Initialize the Vulkan backend
extern bool jitc_vulkan_init();

/// Release all resources held by the Vulkan backend
extern void jitc_vulkan_shutdown();

#endif // defined(DRJIT_ENABLE_VULKAN)
