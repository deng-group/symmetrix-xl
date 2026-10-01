#pragma once

// Shared-buffer staging for Metal modules that read Kokkos host views.
// Slots grow on demand and are reused across launches; read-only tables that
// live for the model lifetime are uploaded once and keyed by host pointer.

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <deque>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "metal_runtime.hpp"

namespace symmetrix::execution::metal {

class MetalStaging {
public:
    MetalStaging(Device device, std::string owner)
        : device_(std::move(device)), owner_(std::move(owner))
    {}

    Buffer& slot(const std::string& name, const std::size_t bytes)
    {
        Buffer& buffer = slots_[name];
        const std::size_t required = std::max<std::size_t>(bytes, 16);
        if (!buffer || buffer.size() < required)
            buffer = device_.allocate(required);
        return buffer;
    }

    template <class T>
    Buffer& upload(const std::string& name, const T* values, const std::size_t count)
    {
        Buffer& buffer = slot(name, count*sizeof(T));
        if (count != 0) {
            if (values == nullptr)
                fail(name+" is null");
            std::memcpy(buffer.contents(), values, count*sizeof(T));
        }
        return buffer;
    }

    // Metal has no FP64; double views are narrowed at the packet boundary.
    Buffer& upload_narrowed(
        const std::string& name, const double* values, const std::size_t count)
    {
        Buffer& buffer = slot(name, count*sizeof(float));
        if (count != 0 && values == nullptr)
            fail(name+" is null");
        float* out = buffer.data<float>();
        for (std::size_t i = 0; i < count; ++i)
            out[i] = static_cast<float>(values[i]);
        return buffer;
    }

    Buffer& zeroed(const std::string& name, const std::size_t count)
    {
        Buffer& buffer = slot(name, count*sizeof(float));
        std::memset(buffer.contents(), 0, buffer.size());
        return buffer;
    }

    // For tables whose host storage is immutable for the model lifetime.
    const Buffer& persistent(const void* host, const std::size_t bytes)
    {
        auto cached = std::find_if(tables_.begin(), tables_.end(),
            [&](const Table& entry) {
                return entry.host == host && entry.bytes == bytes;
            });
        if (cached != tables_.end())
            return cached->buffer;
        if (host == nullptr || bytes == 0)
            fail("persistent table is empty");
        Table entry{host, bytes, device_.allocate(bytes)};
        std::memcpy(entry.buffer.contents(), host, bytes);
        tables_.push_back(std::move(entry));
        return tables_.back().buffer;
    }

    [[noreturn]] void fail(const std::string& message) const
    {
        throw std::runtime_error("Metal "+owner_+": "+message);
    }

private:
    struct Table {
        const void* host = nullptr;
        std::size_t bytes = 0;
        Buffer buffer;
    };

    Device device_;
    std::string owner_;
    std::map<std::string, Buffer> slots_;
    // A deque keeps returned references valid as tables are added.
    std::deque<Table> tables_;
};

}  // namespace symmetrix::execution::metal
