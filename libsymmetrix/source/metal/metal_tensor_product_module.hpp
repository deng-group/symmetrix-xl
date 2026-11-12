#pragma once

// FP32 Metal execution of the MH-1 direct tensor product, the per-edge
// channel-wise sparse Clebsch-Gordan product of the nonlinear backbone used by
// MACE-POLAR-1 and MACE-MH-1. Each (edge, channel) pair is one GPU thread and
// follows the host loop of E3TensorProductKokkosT::try_*_direct_execution
// term by term.

#include <cstddef>
#include <cstdint>
#include <memory>

namespace symmetrix::execution::metal {

class HostMemoryMap;

// The direct-execution plan tables. Pointers refer to host storage that is
// unchanged until release_persistent_tables(); 2-D tables are row-major with
// the given row strides.
struct MetalTensorProductPlan {
    const std::int32_t* instruction_data = nullptr;  // [instructions, stride]
    std::size_t instruction_count = 0;
    std::size_t instruction_stride = 0;
    const float* path_weights = nullptr;  // [instructions]
    const std::int32_t* sparse_indices = nullptr;  // [entries, stride]
    std::size_t sparse_count = 0;
    std::size_t sparse_stride = 0;
    const float* sparse_values = nullptr;  // [entries]
    const std::int32_t* component_offsets = nullptr;
    std::size_t component_offset_count = 0;
    const std::int32_t* harmonic_offsets = nullptr;  // [input_2_dimension+1]
    const std::int32_t* harmonic_terms = nullptr;  // [terms, stride]
    std::size_t harmonic_term_count = 0;
    std::size_t harmonic_term_stride = 0;
    const float* harmonic_values = nullptr;  // [terms]
    const float* output_mask = nullptr;  // [output_dimension]
    std::size_t output_dimension = 0;
    std::size_t input_2_dimension = 0;
    std::uint32_t multiplicity = 0;
};

// One row-major [samples, stride] operand; span is its element count.
struct MetalTensorProductOperand {
    const float* data = nullptr;
    std::size_t stride = 0;
    std::size_t span = 0;
};

struct MetalTensorProductResult {
    float* data = nullptr;
    std::size_t stride = 0;
    std::size_t span = 0;
};

struct MetalTensorProductStatistics {
    std::uint64_t forward_launches = 0;
    std::uint64_t reverse_launches = 0;
    double gpu_seconds = 0.0;
    double staging_seconds = 0.0;
};

class MetalTensorProductModule {
public:
    MetalTensorProductModule();
    ~MetalTensorProductModule();
    MetalTensorProductModule(const MetalTensorProductModule&) = delete;
    MetalTensorProductModule& operator=(const MetalTensorProductModule&) = delete;

    const MetalTensorProductStatistics& statistics() const;
    // Host ranges in this map are read and written in place.
    void set_host_memory(std::shared_ptr<const HostMemoryMap> host_memory);
    void release_persistent_tables();

    // output = sum over instructions of the weighted sparse products; the
    // output is overwritten.
    void forward(
        const MetalTensorProductPlan& plan, std::size_t samples,
        const MetalTensorProductOperand& input_1,
        const MetalTensorProductOperand& input_2,
        const MetalTensorProductOperand& weights,
        const MetalTensorProductResult& output);

    // Adjoints of input_1, input_2, and the weights; all are overwritten.
    void reverse(
        const MetalTensorProductPlan& plan, std::size_t samples,
        const MetalTensorProductOperand& input_1,
        const MetalTensorProductOperand& input_2,
        const MetalTensorProductOperand& weights,
        const MetalTensorProductOperand& output_adjoint,
        const MetalTensorProductResult& input_1_adjoint,
        const MetalTensorProductResult& input_2_adjoint,
        const MetalTensorProductResult& weights_adjoint);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace symmetrix::execution::metal
