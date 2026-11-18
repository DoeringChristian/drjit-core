/*
    src/buffer_map.cpp -- Map device pointers back to the buffer objects
    that contain them

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "buffer_map.h"
#include "common.h"
#include <algorithm>

void BufferMap::ensure_sorted() {
    if (likely(m_sorted))
        return;
    m_entries.erase(
        std::remove_if(m_entries.begin(), m_entries.end(),
                       [](const Entry &e) { return e.handle == nullptr; }),
        m_entries.end());
    std::sort(m_entries.begin(), m_entries.end(),
              [](const Entry &a, const Entry &b) {
                  return a.base < b.base;
              });
    m_sorted = true;
}

void BufferMap::insert(void *ptr, void *handle, size_t size) {
    m_entries.push_back({ (uintptr_t) ptr, handle, size });
    m_sorted = false;
    m_lut.insert_or_assign((uintptr_t) ptr,
                           Entry { (uintptr_t) ptr, handle, size });
}

void *BufferMap::find(void *ptr, size_t *offset_out) {
    uintptr_t addr = (uintptr_t) ptr;

    // Fast path: the pointer references the start of an allocation
    auto it_lut = m_lut.find(addr);
    if (likely(it_lut != m_lut.end())) {
        *offset_out = 0;
        return it_lut->second.handle;
    }

    ensure_sorted();

    auto it = std::upper_bound(
        m_entries.begin(), m_entries.end(), addr,
        [](uintptr_t a, const Entry &b) { return a < b.base; });

    if (it != m_entries.begin()) {
        --it;
        if (it->handle && addr < it->base + it->length) {
            *offset_out = (size_t) (addr - it->base);
            return it->handle;
        }
    }

    *offset_out = 0;
    return nullptr;
}

void *BufferMap::remove(void *ptr) {
    ensure_sorted();

    uintptr_t addr = (uintptr_t) ptr;
    m_lut.erase(addr);

    auto it = std::lower_bound(
        m_entries.begin(), m_entries.end(), addr,
        [](const Entry &e, uintptr_t a) { return e.base < a; });
    if (it == m_entries.end() || it->base != addr)
        return nullptr;
    void *handle = it->handle;
    // Turn the entry into a tombstone to postpone cleanup work
    it->handle = nullptr;
    return handle;
}

void BufferMap::clear() {
    m_entries.clear();
    m_lut.clear();
    m_sorted = true;
}
