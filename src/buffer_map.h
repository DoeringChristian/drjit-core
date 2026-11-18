/*
    src/buffer_map.h -- Map device pointers back to the buffer objects
    that contain them

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "hash.h"
#include <vector>

/**
 * \brief Map from the address range of each allocation to an opaque buffer
 * handle (e.g., an ``id<MTLBuffer>``)
 *
 * Backends whose API operates on buffer objects rather than on addresses use
 * this to resolve a Dr.Jit pointer, which may point into the middle of an
 * allocation, into a (handle, offset) pair. Protected by ``state.lock``.
 */
struct BufferMap {
    /// Register the allocation ``[ptr, ptr + size)`` backed by ``handle``
    void insert(void *ptr, void *handle, size_t size);

    /// Look up the buffer that *contains* the given pointer and the offset.
    /// Returns nullptr if no registered allocation contains ``ptr``.
    void *find(void *ptr, size_t *offset_out);

    /// Unregister the allocation starting at ``ptr`` and return its handle
    void *remove(void *ptr);

    /// Unregister all allocations
    void clear();

private:
    struct Entry {
        uintptr_t base;
        void *handle;
        size_t length;
    };

    void ensure_sorted();

    /// Lazily-sorted flat vector of entries
    std::vector<Entry> m_entries;
    bool m_sorted = true;

    /// Hash table mapping base addresses to entries
    tsl::robin_map<uintptr_t, Entry, UInt64Hasher> m_lut;
};
