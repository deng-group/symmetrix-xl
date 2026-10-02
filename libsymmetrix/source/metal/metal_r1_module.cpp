#include "metal_r1_module.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "metal_r1_abi.hpp"
#include "metal_runtime.hpp"
#include "metal_staging.hpp"

namespace symmetrix::execution::metal {

namespace {

using Clock = std::chrono::steady_clock;

// Below the edge kernel's 384-thread pipeline limit and a multiple of the
// 32-wide Apple SIMD group.
constexpr std::uint64_t threadgroup_size = 128;
constexpr std::uint64_t simd_width = 32;

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error("Metal R1: "+message);
}

bool profiling_enabled()
{
    const char* value = std::getenv("SYMMETRIX_METAL_PROFILE");
    return value != nullptr && std::string(value) == "1";
}

double seconds_since(const Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now()-start).count();
}

std::uint32_t checked_u32(const std::int64_t value, const char* field)
{
    if (value < 0 || value > static_cast<std::int64_t>(UINT32_MAX))
        fail(std::string(field)+" is outside the 32-bit Metal grid range");
    return static_cast<std::uint32_t>(value);
}


}  // namespace

struct MetalR1Module::Impl {
    Device device;
    Pipeline forward;
    Pipeline source;
    Pipeline edge;
    // Present when channels split into 32-wide blocks; one threadgroup per
    // (receiver, block) writes one partial force plane per block.
    Pipeline edge_blocked;
    std::size_t edge_blocks = 0;
    MetalR1Shape shape;
    std::string device_name;
    MetalR1Statistics statistics;
    std::unique_ptr<MetalStaging> staging;

    // GPU buffers whose contents equal a host view written by the last call.
    struct Resident {
        const void* host = nullptr;
        std::size_t count = 0;

        bool matches(const void* pointer, const std::size_t elements) const
        {
            return host != nullptr && host == pointer && count == elements;
        }
    };
    Resident phi1;
    Resident phi1_adjoint;

    static void validate(const MetalA1Layout& layout, MetalStaging& staging)
    {
        if (layout.l_max < 0 || layout.l_max > 3)
            staging.fail("A1 supports l_max from 0 through 3");
    }

    template <class T>
    Buffer& upload(const std::string& name, const T* values, const std::size_t count)
    {
        return staging->upload(name, values, count);
    }

    Buffer& upload_narrowed(
        const std::string& name, const double* values, const std::size_t count)
    {
        return staging->upload_narrowed(name, values, count);
    }

    Buffer& slot(const std::string& name, const std::size_t bytes)
    {
        return staging->slot(name, bytes);
    }

    Buffer& zeroed(const std::string& name, const std::size_t count)
    {
        return staging->zeroed(name, count);
    }

    // The evaluator keeps spline views alive for the model lifetime.
    MetalR1RadialSpline spline(
        const SymmetrixJitHostRadialSplineV2& radial, CommandBatch& batch)
    {
        const std::size_t count = static_cast<std::size_t>(radial.edge_types)
            *radial.intervals*4u*radial.functions;
        const Buffer& coefficients =
            staging->persistent(radial.coefficients, count*sizeof(float));
        batch.use_buffer(coefficients, false);
        return {
            radial.edge_types,
            radial.intervals,
            radial.functions,
            static_cast<float>(radial.h),
            static_cast<float>(radial.x0),
            0,
            coefficients.gpu_address()};
    }
};

MetalR1Module::MetalR1Module(
    const std::string_view msl_source, const MetalR1Shape& shape)
    : impl_(std::make_unique<Impl>())
{
    if (shape.channels <= 0 || shape.edge_harmonics <= 0
            || shape.source_harmonics <= 0 || shape.output_components <= 0)
        fail("module shape must be positive");
    impl_->shape = shape;
    impl_->device = Device::system_default();
    impl_->device_name = impl_->device.information().device_name;
    impl_->staging = std::make_unique<MetalStaging>(impl_->device, "R1");
    const Library library =
        impl_->device.compile(msl_source, module_compile_options());
    impl_->forward = impl_->device.pipeline(library, "symmetrix_r1_forward");
    impl_->source = impl_->device.pipeline(library, "symmetrix_r1_source");
    impl_->edge = impl_->device.pipeline(library, "symmetrix_r1_edge");
    if (library.has_function("symmetrix_r1_edge_blocked")) {
        impl_->edge_blocked =
            impl_->device.pipeline(library, "symmetrix_r1_edge_blocked");
        impl_->edge_blocks = static_cast<std::size_t>(shape.channels)/simd_width;
    }
    for (const Pipeline* pipeline :
            {&impl_->forward, &impl_->source, &impl_->edge}) {
        if (pipeline->max_total_threads_per_threadgroup() < threadgroup_size)
            fail("pipeline '"+pipeline->function_name()
                +"' cannot run the required threadgroup size");
    }
    if (impl_->edge.thread_execution_width() != simd_width)
        fail("the edge kernel requires 32-wide SIMD groups");
}

MetalR1Module::~MetalR1Module() = default;

const MetalR1Shape& MetalR1Module::shape() const { return impl_->shape; }

const std::string& MetalR1Module::device_name() const
{
    return impl_->device_name;
}

const MetalR1Statistics& MetalR1Module::statistics() const
{
    return impl_->statistics;
}

void MetalR1Module::forward(
    const SymmetrixJitHostR1ForwardArgsV2& args,
    const MetalR1ForwardExtents& extents)
{
    if (args.struct_size != sizeof(SymmetrixJitHostR1ForwardArgsV2))
        fail("forward packet size does not match");
    auto& m = *impl_;
    const MetalR1Shape& s = m.shape;
    m.phi1 = {};
    const std::size_t nodes = checked_u32(args.num_nodes, "num_nodes");
    const std::size_t edges = checked_u32(args.num_edges, "num_edges");
    if (nodes == 0)
        return;
    const std::size_t channels = static_cast<std::size_t>(s.channels);
    checked_u32(static_cast<std::int64_t>(nodes*channels), "forward grid");

    const auto staging = Clock::now();
    auto batch = m.device.begin();
    Buffer& node_types = m.upload("node_types", args.node_types, extents.node_types);
    Buffer& num_neigh = m.upload("num_neigh", args.num_neigh, nodes);
    Buffer& first_neigh = m.upload("first_neigh", args.first_neigh, nodes);
    Buffer& neigh_indices = m.upload("neigh_indices", args.neigh_indices, edges);
    Buffer& neigh_types = m.upload("neigh_types", args.neigh_types, edges);
    Buffer& type_to_active =
        m.upload("type_to_active", args.type_to_active, extents.type_to_active);
    Buffer& radius = m.upload_narrowed("radius", args.radius, edges);
    Buffer& harmonics = m.upload("harmonics_values",
        static_cast<const float*>(args.harmonics_values), edges*s.edge_harmonics);
    Buffer& features = m.upload("neighbor_features",
        static_cast<const float*>(args.neighbor_features),
        extents.neighbor_feature_nodes*s.source_harmonics*channels);
    const std::size_t output_count = nodes*s.output_components*channels;
    Buffer& output = m.slot("forward_output", output_count*sizeof(float));

    const MetalR1ForwardArgs packet{
        args.active_type_count,
        static_cast<std::uint32_t>(nodes),
        static_cast<float>(args.cutoff),
        0,
        node_types.gpu_address(),
        num_neigh.gpu_address(),
        first_neigh.gpu_address(),
        neigh_indices.gpu_address(),
        neigh_types.gpu_address(),
        type_to_active.gpu_address(),
        radius.gpu_address(),
        m.spline(args.radial, batch),
        harmonics.gpu_address(),
        features.gpu_address(),
        output.gpu_address()};
    for (Buffer* buffer : {&node_types, &num_neigh, &first_neigh, &neigh_indices,
            &neigh_types, &type_to_active, &radius, &harmonics, &features})
        batch.use_buffer(*buffer, false);
    batch.use_buffer(output, true);
    batch.set_value(0, packet)
        .dispatch_threads(m.forward, {nodes*channels}, {threadgroup_size});
    m.statistics.staging_seconds += seconds_since(staging);

    const double forward_seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += forward_seconds;
    m.statistics.forward_seconds += forward_seconds;
    const auto readback = Clock::now();
    std::memcpy(args.output, output.contents(), output_count*sizeof(float));
    m.phi1 = {args.output, output_count};
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.forward_launches;
}

void MetalR1Module::reverse(
    const SymmetrixJitHostR1SourceArgsV2& source,
    const SymmetrixJitHostR1EdgeArgsV2& edge,
    const MetalR1ReverseExtents& extents)
{
    if (source.struct_size != sizeof(SymmetrixJitHostR1SourceArgsV2)
            || edge.struct_size != sizeof(SymmetrixJitHostR1EdgeArgsV2))
        fail("reverse packet size does not match");
    auto& m = *impl_;
    const MetalR1Shape& s = m.shape;
    const std::size_t sources = checked_u32(source.num_nodes, "source num_nodes");
    const std::size_t edges = checked_u32(edge.num_edges, "num_edges");
    if (static_cast<std::size_t>(source.num_edges) != edges)
        fail("source and edge packets disagree on the edge count");
    if (edges == 0)
        return;
    const std::size_t channels = static_cast<std::size_t>(s.channels);
    checked_u32(static_cast<std::int64_t>(sources*channels), "source grid");
    checked_u32(static_cast<std::int64_t>(edges*simd_width), "edge grid");

    const auto staging = Clock::now();
    auto batch = m.device.begin();
    Buffer& node_types = m.upload("node_types", source.node_types, extents.node_types);
    Buffer& neigh_types = m.upload("neigh_types", source.neigh_types, edges);
    Buffer& type_to_active =
        m.upload("type_to_active", source.type_to_active, extents.type_to_active);
    Buffer& source_offsets =
        m.upload("source_offsets", source.source_offsets, sources+1);
    Buffer& source_edges = m.upload("source_edges", source.source_edges, edges);
    Buffer& edge_receivers = m.upload("edge_receivers", source.edge_receivers, edges);
    Buffer& neigh_indices = m.upload("neigh_indices", edge.neigh_indices, edges);
    Buffer& radius = m.upload_narrowed("radius", source.radius, edges);
    Buffer& harmonics = m.upload("harmonics_values",
        static_cast<const float*>(source.harmonics_values), edges*s.edge_harmonics);
    const std::size_t adjoint_elements =
        extents.receivers*s.output_components*channels;
    Buffer* output_adjoint_buffer = nullptr;
    if (m.phi1_adjoint.matches(source.output_adjoint, adjoint_elements)) {
        output_adjoint_buffer = &m.slot("output_adjoint", adjoint_elements*sizeof(float));
        ++m.statistics.resident_uploads_skipped;
    } else {
        output_adjoint_buffer = &m.upload("output_adjoint",
            static_cast<const float*>(source.output_adjoint), adjoint_elements);
    }
    m.phi1_adjoint = {};
    m.phi1 = {};
    Buffer& output_adjoint = *output_adjoint_buffer;
    Buffer& features = m.upload("neighbor_features",
        static_cast<const float*>(edge.neighbor_features),
        extents.neighbor_feature_nodes*s.source_harmonics*channels);
    // The source owner accumulates into the adjoint, so it starts from the
    // evaluator's current values.
    const std::size_t adjoint_count = sources*s.source_harmonics*channels;
    Buffer& source_adjoint = m.upload("source_adjoint",
        static_cast<const float*>(source.source_adjoint), adjoint_count);

    Buffer& unit_xyz = m.slot("unit_xyz", 3*edges*sizeof(float));
    float* unit = unit_xyz.data<float>();
    if (edge.coordinates_are_unit != 0 && edge.coordinate_scalar_size == sizeof(float)) {
        std::memcpy(unit, edge.xyz, 3*edges*sizeof(float));
    } else if (edge.coordinates_are_unit != 0) {
        const auto* xyz = static_cast<const double*>(edge.xyz);
        for (std::size_t i = 0; i < 3*edges; ++i)
            unit[i] = static_cast<float>(xyz[i]);
    } else {
        const auto* xyz = static_cast<const double*>(edge.xyz);
        for (std::size_t e = 0; e < edges; ++e)
            for (std::size_t k = 0; k < 3; ++k)
                unit[3*e+k] = static_cast<float>(xyz[3*e+k]/edge.radius[e]);
    }
    const bool has_gradients = edge.harmonics_gradients != nullptr;
    Buffer& gradients = has_gradients
        ? m.upload("harmonics_gradients",
            static_cast<const float*>(edge.harmonics_gradients),
            3*edges*s.edge_harmonics)
        : m.slot("harmonics_gradients", 16);
    // Receiver-ordered edges let threadgroups share receiver adjoint rows;
    // otherwise one 32-lane SIMD group owns each edge.
    bool receiver_ordered = static_cast<bool>(m.edge_blocked);
    std::int32_t previous = 0;
    for (std::size_t e = 0; receiver_ordered && e < edges; ++e) {
        const std::int32_t receiver = source.edge_receivers[e];
        receiver_ordered = receiver >= previous
            && static_cast<std::size_t>(receiver) < extents.receivers;
        previous = receiver;
    }
    const std::size_t force_planes = receiver_ordered ? m.edge_blocks : 1;
    Buffer& forces = m.zeroed("directed_forces", force_planes*3*edges);

    const MetalR1RadialSpline radial = m.spline(source.radial, batch);
    const MetalR1SourceArgs source_packet{
        source.active_type_count,
        static_cast<std::uint32_t>(sources),
        static_cast<float>(source.cutoff),
        0,
        node_types.gpu_address(),
        neigh_types.gpu_address(),
        source_offsets.gpu_address(),
        source_edges.gpu_address(),
        edge_receivers.gpu_address(),
        type_to_active.gpu_address(),
        radius.gpu_address(),
        radial,
        harmonics.gpu_address(),
        output_adjoint.gpu_address(),
        source_adjoint.gpu_address()};
    const MetalR1EdgeArgs edge_packet{
        edge.active_type_count,
        static_cast<std::uint32_t>(edges),
        static_cast<float>(edge.cutoff),
        has_gradients ? 1u : 0u,
        node_types.gpu_address(),
        neigh_indices.gpu_address(),
        neigh_types.gpu_address(),
        edge_receivers.gpu_address(),
        type_to_active.gpu_address(),
        unit_xyz.gpu_address(),
        radius.gpu_address(),
        m.spline(edge.radial, batch),
        harmonics.gpu_address(),
        gradients.gpu_address(),
        features.gpu_address(),
        output_adjoint.gpu_address(),
        forces.gpu_address()};
    for (Buffer* buffer : {&node_types, &neigh_types, &type_to_active,
            &source_offsets, &source_edges, &edge_receivers, &neigh_indices,
            &radius, &harmonics, &output_adjoint, &features, &unit_xyz,
            &gradients})
        batch.use_buffer(*buffer, false);
    batch.use_buffer(source_adjoint, true).use_buffer(forces, true);
    if (sources != 0)
        batch.set_value(0, source_packet)
            .dispatch_threads(m.source, {sources*channels}, {threadgroup_size});
    if (profiling_enabled()) {
        const double source_seconds = batch.submit_and_wait().gpu_seconds;
        m.statistics.gpu_seconds += source_seconds;
        m.statistics.source_seconds += source_seconds;
        batch = m.device.begin();
        for (Buffer* buffer : {&node_types, &neigh_types, &type_to_active,
                &edge_receivers, &neigh_indices, &radius, &harmonics,
                &output_adjoint, &features, &unit_xyz, &gradients})
            batch.use_buffer(*buffer, false);
        batch.use_buffer(forces, true);
        m.spline(edge.radial, batch);
    }
    if (receiver_ordered) {
        const std::size_t receiver_count = static_cast<std::size_t>(previous)+1;
        Buffer& offsets = m.slot("receiver_offsets",
            (receiver_count+1)*sizeof(std::int32_t));
        auto* offset = offsets.data<std::int32_t>();
        std::fill(offset, offset+receiver_count+1, 0);
        for (std::size_t e = 0; e < edges; ++e)
            ++offset[source.edge_receivers[e]+1];
        for (std::size_t r = 0; r < receiver_count; ++r)
            offset[r+1] += offset[r];
        batch.set_value(0, edge_packet).set_buffer(1, offsets).set_buffer(2, forces)
            .dispatch_threadgroups(m.edge_blocked,
                {receiver_count*m.edge_blocks}, {threadgroup_size});
        ++m.statistics.blocked_edge_launches;
    } else {
        batch.set_value(0, edge_packet)
            .dispatch_threads(m.edge, {edges*simd_width}, {threadgroup_size});
    }
    m.statistics.staging_seconds += seconds_since(staging);

    const double edge_seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += edge_seconds;
    if (profiling_enabled())
        m.statistics.edge_seconds += edge_seconds;
    const auto readback = Clock::now();
    std::memcpy(source.source_adjoint, source_adjoint.contents(),
        adjoint_count*sizeof(float));
    // The edge kernel subtracts from zero; the evaluator's directed forces
    // receive the same contribution in double precision.
    const float* force_values = forces.data<float>();
    if (force_planes == 1)
        for (std::size_t i = 0; i < 3*edges; ++i)
            edge.directed_forces[i] += static_cast<double>(force_values[i]);
    else
        for (std::size_t i = 0; i < 3*edges; ++i) {
            double sum = 0.0;
            for (std::size_t plane = 0; plane < force_planes; ++plane)
                sum += static_cast<double>(force_values[plane*3*edges+i]);
            edge.directed_forces[i] += sum;
        }
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.reverse_launches;
}

// For degree l and row m, the Phi1 rows of all nodes form a strided matrix
// with one row per node, so A1 is 2l+1 GEMMs per degree over every node.
bool MetalR1Module::a1_forward(
    const float* phi1, const std::size_t num_nodes,
    const MetalA1Layout& layout, float* a1)
{
    auto& m = *impl_;
    Impl::validate(layout, *m.staging);
    const std::size_t channels = static_cast<std::size_t>(m.shape.channels);
    const std::size_t phi1_count = num_nodes*layout.num_lme*channels;
    if (num_nodes == 0 || !m.phi1.matches(phi1, phi1_count))
        return false;
    checked_u32(static_cast<std::int64_t>(num_nodes), "A1 node count");
    const auto rows = static_cast<std::uint32_t>(num_nodes);
    const auto c = static_cast<std::uint32_t>(channels);
    auto batch = m.device.begin();
    Buffer& input = m.slot("forward_output", phi1_count*sizeof(float));
    const std::size_t output_count = num_nodes*layout.num_lm*channels;
    Buffer& output = m.slot("a1_output", output_count*sizeof(float));
    const std::size_t input_row = layout.num_lme*channels*sizeof(float);
    const std::size_t output_row = layout.num_lm*channels*sizeof(float);
    for (int l = 0; l <= layout.l_max; ++l) {
        const auto inputs = static_cast<std::uint32_t>(layout.eta[l]*channels);
        if (inputs == 0)
            continue;
        const Buffer& weights = m.staging->persistent(
            layout.weights[l], static_cast<std::size_t>(inputs)*channels*sizeof(float));
        for (int row = 0; row < 2*l+1; ++row)
            batch.gemm(
                {&input, (layout.lme[l]*channels+static_cast<std::size_t>(row)*inputs)
                    *sizeof(float), rows, inputs, input_row},
                {&weights, 0, inputs, c, channels*sizeof(float)},
                {&output, (static_cast<std::size_t>(l*l+row)*channels)*sizeof(float),
                    rows, c, output_row});
    }
    const double a1_seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += a1_seconds;
    m.statistics.a1_seconds += a1_seconds;
    const auto readback = Clock::now();
    std::memcpy(a1, output.contents(), output_count*sizeof(float));
    // Degrees without paths contribute zero rows, as the zero-depth host GEMM.
    for (int l = 0; l <= layout.l_max; ++l)
        if (layout.eta[l] == 0)
            for (std::size_t node = 0; node < num_nodes; ++node)
                std::memset(a1+(node*layout.num_lm+l*l)*channels, 0,
                    (2*l+1)*channels*sizeof(float));
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.a1_forward_launches;
    return true;
}

void MetalR1Module::a1_reverse(
    const std::size_t num_nodes, const std::size_t capacity_nodes,
    const MetalA1Layout& layout,
    const float* a1_adjoint, float* phi1_adjoint)
{
    auto& m = *impl_;
    Impl::validate(layout, *m.staging);
    if (num_nodes == 0)
        return;
    if (capacity_nodes < num_nodes)
        m.staging->fail("dPhi1 capacity is smaller than the node count");
    checked_u32(static_cast<std::int64_t>(num_nodes), "A1 node count");
    const std::size_t channels = static_cast<std::size_t>(m.shape.channels);
    const auto rows = static_cast<std::uint32_t>(num_nodes);
    const auto c = static_cast<std::uint32_t>(channels);
    const auto staging = Clock::now();
    auto batch = m.device.begin();
    Buffer& input = m.upload("a1_adjoint", a1_adjoint,
        num_nodes*layout.num_lm*channels);
    const std::size_t output_count = num_nodes*layout.num_lme*channels;
    const std::size_t capacity_count = capacity_nodes*layout.num_lme*channels;
    // Written into the R1 reverse input slot, sized as the reverse stages
    // it, so the next reverse reuses it without a reallocation.
    Buffer& output = m.slot("output_adjoint", capacity_count*sizeof(float));
    const std::size_t input_row = layout.num_lm*channels*sizeof(float);
    const std::size_t output_row = layout.num_lme*channels*sizeof(float);
    for (int l = 0; l <= layout.l_max; ++l) {
        const auto inputs = static_cast<std::uint32_t>(layout.eta[l]*channels);
        if (inputs == 0)
            continue;
        const Buffer& weights_trans = m.staging->persistent(
            layout.weights_trans[l],
            static_cast<std::size_t>(inputs)*channels*sizeof(float));
        for (int row = 0; row < 2*l+1; ++row)
            batch.gemm(
                {&input, (static_cast<std::size_t>(l*l+row)*channels)*sizeof(float),
                    rows, c, input_row},
                {&weights_trans, 0, c, inputs, inputs*sizeof(float)},
                {&output, (layout.lme[l]*channels+static_cast<std::size_t>(row)*inputs)
                    *sizeof(float), rows, inputs, output_row});
    }
    m.statistics.staging_seconds += seconds_since(staging);
    const double a1_seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += a1_seconds;
    m.statistics.a1_seconds += a1_seconds;
    const auto readback = Clock::now();
    std::memcpy(phi1_adjoint, output.contents(), output_count*sizeof(float));
    m.phi1_adjoint = {phi1_adjoint, capacity_count};
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.a1_reverse_launches;
}

}  // namespace symmetrix::execution::metal
