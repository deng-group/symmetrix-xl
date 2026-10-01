#include "metal_r0_module.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

#include "metal_runtime.hpp"
#include "metal_staging.hpp"

namespace symmetrix::execution::metal {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t threadgroup_size = 128;
constexpr std::uint64_t simd_width = 32;
constexpr std::int32_t max_harmonics = 16;

// Device pointers are 64-bit GPU addresses; the scalar header is 48 bytes so
// the pointer block starts 8-byte aligned in both layouts.
struct R0Args {
    std::uint32_t num_nodes;
    std::uint32_t active_type_count;
    std::uint32_t channels;
    std::uint32_t harmonic_count;
    std::uint32_t intervals;
    std::uint32_t functions;
    std::uint32_t num_edges;
    std::uint32_t apply_scale;
    float h;
    float x0;
    float cutoff;
    std::uint32_t reserved;
    std::uint64_t node_types;
    std::uint64_t num_neigh;
    std::uint64_t first_neigh;
    std::uint64_t neigh_types;
    std::uint64_t type_to_active;
    std::uint64_t edge_receivers;
    std::uint64_t radius;
    std::uint64_t coefficients;
    std::uint64_t harmonics;
    std::uint64_t density_scale;
    std::uint64_t unit_xyz;
    std::uint64_t gradients;
    std::uint64_t adjoint;
    std::uint64_t output;
};
static_assert(sizeof(R0Args) == 160);

constexpr const char* r0_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct R0Args {
    uint num_nodes;
    uint active_type_count;
    uint channels;
    uint harmonic_count;
    uint intervals;
    uint functions;
    uint num_edges;
    uint apply_scale;
    float h;
    float x0;
    float cutoff;
    uint reserved;
    device const int* node_types;
    device const int* num_neigh;
    device const int* first_neigh;
    device const int* neigh_types;
    device const int* type_to_active;
    device const int* edge_receivers;
    device const float* radius;
    device const float* coefficients;
    device const float* harmonics;
    device const float* density_scale;
    device const float* unit_xyz;
    device const float* gradients;
    device const float* adjoint;
    device float* output;
};

struct SplinePoint {
    ulong base;
    float x;
    float xx;
    float xxx;
};

inline SplinePoint spline_point(constant R0Args& a, int edge_type, float r)
{
    int interval = int(floor((r - a.x0) / a.h));
    float x = r - a.x0 - a.h * float(interval);
    if (interval < 0) {
        interval = 0;
        x = 0.0f;
    } else if (interval >= int(a.intervals)) {
        interval = int(a.intervals) - 1;
        x = a.h;
    }
    const float xx = x * x;
    return {(ulong(edge_type) * a.intervals + ulong(interval)) * 4u * a.functions,
            x, xx, xx * x};
}

inline uint harmonic_degree_count(uint harmonic_count)
{
    uint l = 0;
    while ((l + 1) * (l + 1) < harmonic_count)
        ++l;
    return l + 1;
}

// One thread owns one (receiver, channel) output row of A0.
kernel void symmetrix_r0_forward(
    constant R0Args& a [[buffer(0)]],
    uint flat [[thread_position_in_grid]])
{
    const uint receiver = flat / a.channels;
    if (receiver >= a.num_nodes)
        return;
    const uint channel = flat % a.channels;
    const uint degrees = harmonic_degree_count(a.harmonic_count);
    const int receiver_type = a.type_to_active[a.node_types[receiver]];
    const int begin = a.first_neigh[receiver];
    const int end = begin + a.num_neigh[receiver];
    float accumulator[16];
    for (uint lm = 0; lm < 16; ++lm)
        accumulator[lm] = 0.0f;
    for (int edge = begin; edge < end; ++edge) {
        const float r = a.radius[edge];
        if (!(r < a.cutoff))
            continue;
        const int neighbor_type = a.type_to_active[a.neigh_types[edge]];
        const SplinePoint p = spline_point(
            a, receiver_type * int(a.active_type_count) + neighbor_type, r);
        const ulong harmonic_offset = ulong(edge) * a.harmonic_count;
        for (uint l = 0; l < degrees; ++l) {
            const ulong index = p.base + l * a.channels + channel;
            const float value = a.coefficients[index]
                + a.coefficients[index + a.functions] * p.x
                + a.coefficients[index + 2 * a.functions] * p.xx
                + a.coefficients[index + 3 * a.functions] * p.xxx;
            for (uint lm = l * l; lm < (l + 1) * (l + 1); ++lm)
                accumulator[lm] += value * a.harmonics[harmonic_offset + lm];
        }
    }
    const float scale = a.apply_scale != 0 ? a.density_scale[receiver] : 1.0f;
    for (uint lm = 0; lm < a.harmonic_count; ++lm)
        a.output[(ulong(receiver) * a.harmonic_count + lm) * a.channels + channel] =
            accumulator[lm] * scale;
}

// One 32-lane SIMD group owns one edge; lanes split the channels so adjoint
// and spline loads coalesce, and simd_sum reduces the force components.
kernel void symmetrix_r0_reverse(
    constant R0Args& a [[buffer(0)]],
    uint flat [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint edge = flat / 32u;
    if (edge >= a.num_edges)
        return;
    const int receiver = a.edge_receivers[edge];
    const float r = a.radius[edge];
    if (receiver < 0 || !(r < a.cutoff))
        return;
    const uint degrees = harmonic_degree_count(a.harmonic_count);
    const int receiver_type = a.type_to_active[a.node_types[receiver]];
    const int neighbor_type = a.type_to_active[a.neigh_types[edge]];
    const SplinePoint p = spline_point(
        a, receiver_type * int(a.active_type_count) + neighbor_type, r);
    const float ux = a.unit_xyz[3 * edge];
    const float uy = a.unit_xyz[3 * edge + 1];
    const float uz = a.unit_xyz[3 * edge + 2];
    const ulong harmonic_offset = ulong(edge) * a.harmonic_count;
    const ulong gradient_offset = 3 * harmonic_offset;
    const ulong adjoint_offset = ulong(receiver) * a.harmonic_count;
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    for (uint channel = lane; channel < a.channels; channel += 32u) {
        for (uint l = 0; l < degrees; ++l) {
            const ulong index = p.base + l * a.channels + channel;
            const float c1 = a.coefficients[index + a.functions];
            const float c2 = a.coefficients[index + 2 * a.functions];
            const float c3 = a.coefficients[index + 3 * a.functions];
            const float value = a.coefficients[index] + c1 * p.x + c2 * p.xx + c3 * p.xxx;
            const float derivative = c1 + 2.0f * c2 * p.x + 3.0f * c3 * p.xx;
            for (uint lm = l * l; lm < (l + 1) * (l + 1); ++lm) {
                const float adjoint =
                    a.adjoint[(adjoint_offset + lm) * a.channels + channel];
                const float radial = derivative * a.harmonics[harmonic_offset + lm] * adjoint;
                const float angular = value * adjoint;
                fx += radial * ux + angular * a.gradients[gradient_offset + lm];
                fy += radial * uy
                    + angular * a.gradients[gradient_offset + a.harmonic_count + lm];
                fz += radial * uz
                    + angular * a.gradients[gradient_offset + 2 * a.harmonic_count + lm];
            }
        }
    }
    fx = simd_sum(fx);
    fy = simd_sum(fy);
    fz = simd_sum(fz);
    if (lane == 0) {
        a.output[3 * edge] = -fx;
        a.output[3 * edge + 1] = -fy;
        a.output[3 * edge + 2] = -fz;
    }
}
)MSL";

double seconds_since(const Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now()-start).count();
}

}  // namespace

struct MetalR0Module::Impl {
    Device device;
    Pipeline forward;
    Pipeline reverse;
    std::unique_ptr<MetalStaging> staging;
    MetalR0Statistics statistics;

    [[noreturn]] void fail(const std::string& message) const
    {
        staging->fail(message);
    }

    void validate(const MetalR0Graph& graph, const MetalR0Spline& spline,
        const std::int32_t channels, const std::int32_t harmonic_count) const
    {
        if (graph.num_nodes < 0 || channels <= 0)
            fail("graph extents must be nonnegative and channels positive");
        if (harmonic_count <= 0 || harmonic_count > max_harmonics)
            fail("R0 supports l_max from 0 through 3");
        if (spline.coefficients == nullptr || spline.intervals == 0
                || spline.functions == 0 || spline.edge_types == 0)
            fail("R0 spline coefficients are empty");
        const std::size_t degrees = [&] {
            std::size_t l = 0;
            while ((l+1)*(l+1) < static_cast<std::size_t>(harmonic_count))
                ++l;
            return l+1;
        }();
        if (spline.functions != degrees*static_cast<std::size_t>(channels))
            fail("R0 spline functions do not match the channel layout");
        if (static_cast<std::size_t>(graph.active_type_count)
                *static_cast<std::size_t>(graph.active_type_count)
                > spline.edge_types)
            fail("R0 spline edge types do not cover the active types");
        if (graph.edge_capacity > UINT32_MAX
                || static_cast<std::size_t>(graph.num_nodes)*channels > UINT32_MAX
                || graph.edge_capacity*simd_width > UINT32_MAX)
            fail("R0 graph exceeds the 32-bit Metal grid range");
    }

    // Uploads the receiver-ordered graph shared by both passes.
    void stage_graph(const MetalR0Graph& graph, const MetalR0Spline& spline,
        CommandBatch& batch, R0Args& args)
    {
        const std::size_t nodes = static_cast<std::size_t>(graph.num_nodes);
        const std::size_t edges = graph.edge_capacity;
        Buffer& node_types =
            staging->upload("node_types", graph.node_types, graph.node_type_count);
        Buffer& num_neigh = staging->upload("num_neigh", graph.num_neigh, nodes);
        Buffer& first_neigh = staging->upload("first_neigh", graph.first_neigh, nodes);
        Buffer& neigh_types = staging->upload("neigh_types", graph.neigh_types, edges);
        Buffer& type_to_active = staging->upload(
            "type_to_active", graph.type_to_active, graph.type_map_count);
        Buffer& radius = staging->upload_narrowed("radius", graph.radius, edges);
        const Buffer& coefficients = staging->persistent(spline.coefficients,
            spline.edge_types*spline.intervals*4*spline.functions*sizeof(float));
        for (const Buffer* buffer : std::initializer_list<const Buffer*>{
                &node_types, &num_neigh, &first_neigh, &neigh_types,
                &type_to_active, &radius, &coefficients})
            batch.use_buffer(*buffer, false);
        args.num_nodes = static_cast<std::uint32_t>(nodes);
        args.active_type_count = static_cast<std::uint32_t>(graph.active_type_count);
        args.intervals = static_cast<std::uint32_t>(spline.intervals);
        args.functions = static_cast<std::uint32_t>(spline.functions);
        args.num_edges = static_cast<std::uint32_t>(edges);
        args.h = static_cast<float>(spline.h);
        args.x0 = static_cast<float>(spline.x0);
        args.cutoff = static_cast<float>(graph.cutoff);
        args.node_types = node_types.gpu_address();
        args.num_neigh = num_neigh.gpu_address();
        args.first_neigh = first_neigh.gpu_address();
        args.neigh_types = neigh_types.gpu_address();
        args.type_to_active = type_to_active.gpu_address();
        args.radius = radius.gpu_address();
        args.coefficients = coefficients.gpu_address();
    }
};

MetalR0Module::MetalR0Module() : impl_(std::make_unique<Impl>())
{
    impl_->device = Device::system_default();
    impl_->staging = std::make_unique<MetalStaging>(impl_->device, "R0");
    const Library library = impl_->device.compile(r0_source);
    impl_->forward = impl_->device.pipeline(library, "symmetrix_r0_forward");
    impl_->reverse = impl_->device.pipeline(library, "symmetrix_r0_reverse");
    if (impl_->reverse.thread_execution_width() != simd_width)
        impl_->fail("the reverse kernel requires 32-wide SIMD groups");
    for (const Pipeline* pipeline : {&impl_->forward, &impl_->reverse})
        if (pipeline->max_total_threads_per_threadgroup() < threadgroup_size)
            impl_->fail("pipeline '"+pipeline->function_name()
                +"' cannot run the required threadgroup size");
}

MetalR0Module::~MetalR0Module() = default;

const MetalR0Statistics& MetalR0Module::statistics() const
{
    return impl_->statistics;
}

void MetalR0Module::forward(
    const MetalR0Graph& graph,
    const MetalR0Spline& spline,
    const std::int32_t channels,
    const std::int32_t harmonic_count,
    const float* harmonics,
    const double* density_scale,
    float* output)
{
    auto& m = *impl_;
    m.validate(graph, spline, channels, harmonic_count);
    const std::size_t nodes = static_cast<std::size_t>(graph.num_nodes);
    if (nodes == 0)
        return;
    const auto staging = Clock::now();
    auto batch = m.device.begin();
    R0Args args{};
    m.stage_graph(graph, spline, batch, args);
    args.channels = static_cast<std::uint32_t>(channels);
    args.harmonic_count = static_cast<std::uint32_t>(harmonic_count);
    args.apply_scale = density_scale != nullptr ? 1u : 0u;
    Buffer& Y = m.staging->upload("harmonics", harmonics,
        graph.edge_capacity*static_cast<std::size_t>(harmonic_count));
    Buffer& scale = density_scale != nullptr
        ? m.staging->upload_narrowed("density_scale", density_scale, nodes)
        : m.staging->slot("density_scale", 16);
    const std::size_t output_count =
        nodes*static_cast<std::size_t>(harmonic_count)*channels;
    Buffer& out = m.staging->slot("forward_output", output_count*sizeof(float));
    args.harmonics = Y.gpu_address();
    args.density_scale = scale.gpu_address();
    args.output = out.gpu_address();
    batch.use_buffer(Y, false).use_buffer(scale, false).use_buffer(out, true);
    batch.set_value(0, args)
        .dispatch_threads(m.forward, {nodes*channels}, {threadgroup_size});
    m.statistics.staging_seconds += seconds_since(staging);
    m.statistics.gpu_seconds += batch.submit_and_wait().gpu_seconds;
    const auto readback = Clock::now();
    std::memcpy(output, out.contents(), output_count*sizeof(float));
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.forward_launches;
}

void MetalR0Module::coordinate_reverse(
    const MetalR0Graph& graph,
    const MetalR0Spline& spline,
    const std::int32_t channels,
    const std::int32_t harmonic_count,
    const void* coordinates,
    const std::uint32_t coordinate_scalar_size,
    const bool coordinates_are_unit,
    const float* harmonics,
    const float* harmonic_gradients,
    const float* output_adjoint,
    double* directed_forces)
{
    auto& m = *impl_;
    m.validate(graph, spline, channels, harmonic_count);
    if (harmonic_gradients == nullptr)
        m.fail("R0 coordinate reverse requires harmonic gradients");
    const std::size_t nodes = static_cast<std::size_t>(graph.num_nodes);
    const std::size_t edges = graph.edge_capacity;
    if (nodes == 0 || edges == 0)
        return;
    const auto staging = Clock::now();
    auto batch = m.device.begin();
    R0Args args{};
    m.stage_graph(graph, spline, batch, args);
    args.channels = static_cast<std::uint32_t>(channels);
    args.harmonic_count = static_cast<std::uint32_t>(harmonic_count);

    // Receivers of the edges they own; edges outside every receiver range
    // are skipped by the kernel.
    Buffer& receivers = m.staging->slot("edge_receivers", edges*sizeof(std::int32_t));
    auto* receiver_of = receivers.data<std::int32_t>();
    std::fill(receiver_of, receiver_of+edges, -1);
    for (std::size_t node = 0; node < nodes; ++node) {
        const std::int64_t begin = graph.first_neigh[node];
        const std::int64_t end = begin+graph.num_neigh[node];
        if (begin < 0 || end > static_cast<std::int64_t>(edges))
            m.fail("receiver edge range exceeds the edge capacity");
        for (std::int64_t edge = begin; edge < end; ++edge)
            receiver_of[edge] = static_cast<std::int32_t>(node);
    }
    Buffer& unit = m.staging->slot("unit_xyz", 3*edges*sizeof(float));
    float* unit_xyz = unit.data<float>();
    if (coordinates_are_unit && coordinate_scalar_size == sizeof(float)) {
        std::memcpy(unit_xyz, coordinates, 3*edges*sizeof(float));
    } else {
        const auto* xyz = static_cast<const double*>(coordinates);
        for (std::size_t edge = 0; edge < edges; ++edge)
            for (std::size_t k = 0; k < 3; ++k)
                unit_xyz[3*edge+k] = static_cast<float>(coordinates_are_unit
                    ? xyz[3*edge+k] : xyz[3*edge+k]/graph.radius[edge]);
    }
    const std::size_t harmonic_values = edges*static_cast<std::size_t>(harmonic_count);
    Buffer& Y = m.staging->upload("harmonics", harmonics, harmonic_values);
    Buffer& gradients =
        m.staging->upload("gradients", harmonic_gradients, 3*harmonic_values);
    Buffer& adjoint = m.staging->upload("adjoint", output_adjoint,
        nodes*static_cast<std::size_t>(harmonic_count)*channels);
    Buffer& forces = m.staging->zeroed("forces", 3*edges);
    args.edge_receivers = receivers.gpu_address();
    args.unit_xyz = unit.gpu_address();
    args.harmonics = Y.gpu_address();
    args.gradients = gradients.gpu_address();
    args.adjoint = adjoint.gpu_address();
    args.output = forces.gpu_address();
    for (const Buffer* buffer : std::initializer_list<const Buffer*>{
            &receivers, &unit, &Y, &gradients, &adjoint})
        batch.use_buffer(*buffer, false);
    batch.use_buffer(forces, true);
    batch.set_value(0, args)
        .dispatch_threads(m.reverse, {edges*simd_width}, {threadgroup_size});
    m.statistics.staging_seconds += seconds_since(staging);
    m.statistics.gpu_seconds += batch.submit_and_wait().gpu_seconds;
    const auto readback = Clock::now();
    const float* values = forces.data<float>();
    for (std::size_t i = 0; i < 3*edges; ++i)
        directed_forces[i] += static_cast<double>(values[i]);
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.reverse_launches;
}

}  // namespace symmetrix::execution::metal
