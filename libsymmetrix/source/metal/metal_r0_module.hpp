#pragma once

// FP32 Metal execution of the standard first-interaction R0 module. The
// receiver-ordered edge graph and FP32 tensors come from Kokkos host views;
// the density-scale force term, which evaluates a double-precision spline,
// stays with the caller.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace symmetrix::execution::metal {

struct MetalR0Graph {
    std::int32_t num_nodes = 0;
    std::int32_t active_type_count = 0;
    double cutoff = 0.0;
    const std::int32_t* node_types = nullptr;
    std::size_t node_type_count = 0;
    const std::int32_t* num_neigh = nullptr;
    const std::int32_t* first_neigh = nullptr;
    const std::int32_t* neigh_types = nullptr;
    const std::int32_t* type_to_active = nullptr;
    std::size_t type_map_count = 0;
    // Indexed by absolute edge; covers every edge of the receivers.
    const double* radius = nullptr;
    std::size_t edge_capacity = 0;
};

struct MetalR0Spline {
    double h = 0.0;
    double x0 = 0.0;
    // [edge_types, intervals, 4, functions], functions = (l_max+1)*channels.
    const float* coefficients = nullptr;
    std::size_t edge_types = 0;
    std::size_t intervals = 0;
    std::size_t functions = 0;
};

struct MetalR0Statistics {
    std::uint64_t forward_launches = 0;
    std::uint64_t reverse_launches = 0;
    double gpu_seconds = 0.0;
    double staging_seconds = 0.0;
};

class MetalR0Module {
public:
    MetalR0Module();
    ~MetalR0Module();
    MetalR0Module(const MetalR0Module&) = delete;
    MetalR0Module& operator=(const MetalR0Module&) = delete;

    const MetalR0Statistics& statistics() const;

    // output[receiver, lm, channel] = density_scale(receiver)
    //     * sum_edges R_{l(lm), channel}(r) Y_lm.
    void forward(
        const MetalR0Graph& graph,
        const MetalR0Spline& spline,
        std::int32_t channels,
        std::int32_t harmonic_count,
        const float* harmonics,
        const double* density_scale,
        float* output);

    // Adds the radial and angular coordinate adjoint of every edge to
    // directed_forces[3*edge + k]. Unit directions follow the evaluator's
    // geometry policy: FP32 unit vectors, FP64 unit vectors, or FP64 vectors.
    void coordinate_reverse(
        const MetalR0Graph& graph,
        const MetalR0Spline& spline,
        std::int32_t channels,
        std::int32_t harmonic_count,
        const void* coordinates,
        std::uint32_t coordinate_scalar_size,
        bool coordinates_are_unit,
        const float* harmonics,
        const float* harmonic_gradients,
        const float* output_adjoint,
        double* directed_forces);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace symmetrix::execution::metal
