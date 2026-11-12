#include "metal_tensor_product_module.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <string>

#include "metal_runtime.hpp"
#include "metal_staging.hpp"

namespace symmetrix::execution::metal {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now()-start).count();
}

// Shared with the kernels below; 64-bit fields first so that the host and
// Metal layouts agree.
struct TensorProductArgs {
    std::uint64_t stride_1 = 0;
    std::uint64_t stride_2 = 0;
    std::uint64_t stride_weights = 0;
    std::uint64_t stride_output = 0;
    std::uint32_t samples = 0;
    std::uint32_t multiplicity = 0;
    std::uint32_t instruction_count = 0;
    std::uint32_t instruction_stride = 0;
    std::uint32_t sparse_stride = 0;
    std::uint32_t harmonic_term_stride = 0;
    std::uint32_t input_2_dimension = 0;
    std::uint32_t padding = 0;
};

// The loops and operand order follow the host kernels in
// mace_nonlinear_mh1_fast_path.cpp so that FP32 results agree to rounding.
constexpr const char* kernel_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct TensorProductArgs {
    ulong stride_1;
    ulong stride_2;
    ulong stride_weights;
    ulong stride_output;
    uint samples;
    uint multiplicity;
    uint instruction_count;
    uint instruction_stride;
    uint sparse_stride;
    uint harmonic_term_stride;
    uint input_2_dimension;
    uint padding;
};

kernel void symmetrix_tp_zero(
    device float* data [[buffer(0)]],
    constant ulong& count [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count)
        data[index] = 0.0f;
}

kernel void symmetrix_tp_forward(
    device const int* plan [[buffer(0)]],
    device const float* paths [[buffer(1)]],
    device const int* indices [[buffer(2)]],
    device const float* values [[buffer(3)]],
    device const int* component_offsets [[buffer(4)]],
    device const float* mask [[buffer(5)]],
    device const float* input_1 [[buffer(6)]],
    device const float* input_2 [[buffer(7)]],
    device const float* weights [[buffer(8)]],
    device float* output [[buffer(9)]],
    constant TensorProductArgs& args [[buffer(10)]],
    uint2 position [[thread_position_in_grid]])
{
    const uint channel = position.x;
    const uint sample = position.y;
    if (channel >= args.multiplicity || sample >= args.samples)
        return;
    device const float* first = input_1+ulong(sample)*args.stride_1;
    device const float* second = input_2+ulong(sample)*args.stride_2;
    device const float* weight = weights+ulong(sample)*args.stride_weights;
    device float* out = output+ulong(sample)*args.stride_output;
    for (uint instruction = 0; instruction < args.instruction_count; ++instruction) {
        device const int* row = plan+instruction*args.instruction_stride;
        const int input_1_offset = row[0];
        const int input_2_offset = row[1];
        const int output_offset = row[2];
        const int input_width = row[3];
        const int output_width = row[4];
        const int weight_offset = row[5];
        const int component_base = row[6];
        const float scale = paths[instruction]*weight[weight_offset+int(channel)];
        for (int component = 0; component < output_width; ++component) {
            float result = 0.0f;
            for (int entry = component_offsets[component_base+component];
                    entry < component_offsets[component_base+component+1]; ++entry) {
                device const int* index = indices+entry*int(args.sparse_stride);
                result += values[entry]
                    *first[input_1_offset+int(channel)*input_width+index[0]]
                    *second[input_2_offset+index[1]];
            }
            const int output_index = output_offset+int(channel)*output_width+component;
            out[output_index] += scale*result*mask[output_index];
        }
    }
}

kernel void symmetrix_tp_reverse_channels(
    device const int* plan [[buffer(0)]],
    device const float* paths [[buffer(1)]],
    device const int* indices [[buffer(2)]],
    device const float* values [[buffer(3)]],
    device const int* component_offsets [[buffer(4)]],
    device const float* mask [[buffer(5)]],
    device const float* input_1 [[buffer(6)]],
    device const float* input_2 [[buffer(7)]],
    device const float* weights [[buffer(8)]],
    device const float* output_adjoint [[buffer(9)]],
    device float* input_1_adjoint [[buffer(10)]],
    device float* weights_adjoint [[buffer(11)]],
    constant TensorProductArgs& args [[buffer(12)]],
    uint2 position [[thread_position_in_grid]])
{
    const uint channel = position.x;
    const uint sample = position.y;
    if (channel >= args.multiplicity || sample >= args.samples)
        return;
    device const float* first = input_1+ulong(sample)*args.stride_1;
    device const float* second = input_2+ulong(sample)*args.stride_2;
    device const float* weight = weights+ulong(sample)*args.stride_weights;
    device const float* adjoint = output_adjoint+ulong(sample)*args.stride_output;
    device float* first_adjoint = input_1_adjoint+ulong(sample)*args.stride_1;
    device float* weight_adjoint = weights_adjoint+ulong(sample)*args.stride_weights;
    for (uint instruction = 0; instruction < args.instruction_count; ++instruction) {
        device const int* row = plan+instruction*args.instruction_stride;
        const int input_1_offset = row[0];
        const int input_2_offset = row[1];
        const int output_offset = row[2];
        const int input_width = row[3];
        const int output_width = row[4];
        const int weight_offset = row[5];
        const int component_base = row[6];
        const int first_entry = component_offsets[component_base];
        const int last_entry = component_offsets[component_base+output_width];
        const float w = weight[weight_offset+int(channel)];
        float weight_value = 0.0f;
        for (int entry = first_entry; entry < last_entry; ++entry) {
            device const int* index = indices+entry*int(args.sparse_stride);
            const int a = index[0];
            const int b = index[1];
            const int c = index[2];
            const int output_index = output_offset+int(channel)*output_width+c;
            const float common = paths[instruction]*values[entry]
                *mask[output_index]*adjoint[output_index];
            const int input_index = input_1_offset+int(channel)*input_width+a;
            const float x = first[input_index];
            const float y = second[input_2_offset+b];
            first_adjoint[input_index] += common*w*y;
            weight_value += common*x*y;
        }
        weight_adjoint[weight_offset+int(channel)] = weight_value;
    }
}

kernel void symmetrix_tp_reverse_harmonics(
    device const int* plan [[buffer(0)]],
    device const float* paths [[buffer(1)]],
    device const int* harmonic_offsets [[buffer(2)]],
    device const int* harmonic_terms [[buffer(3)]],
    device const float* harmonic_values [[buffer(4)]],
    device const float* mask [[buffer(5)]],
    device const float* input_1 [[buffer(6)]],
    device const float* weights [[buffer(7)]],
    device const float* output_adjoint [[buffer(8)]],
    device float* input_2_adjoint [[buffer(9)]],
    constant TensorProductArgs& args [[buffer(10)]],
    uint2 position [[thread_position_in_grid]])
{
    const uint component = position.x;
    const uint sample = position.y;
    if (component >= args.input_2_dimension || sample >= args.samples)
        return;
    device const float* first = input_1+ulong(sample)*args.stride_1;
    device const float* weight = weights+ulong(sample)*args.stride_weights;
    device const float* adjoint = output_adjoint+ulong(sample)*args.stride_output;
    float result = 0.0f;
    for (uint channel = 0; channel < args.multiplicity; ++channel)
        for (int term = harmonic_offsets[component];
                term < harmonic_offsets[component+1]; ++term) {
            device const int* entry = harmonic_terms+term*int(args.harmonic_term_stride);
            const int instruction = entry[0];
            const int a = entry[1];
            const int c = entry[2];
            device const int* row = plan+instruction*int(args.instruction_stride);
            const int input_1_offset = row[0];
            const int output_offset = row[2];
            const int input_width = row[3];
            const int output_width = row[4];
            const int weight_offset = row[5];
            const int output_index = output_offset+int(channel)*output_width+c;
            const float common = paths[instruction]*harmonic_values[term]
                *mask[output_index]*adjoint[output_index];
            result += common*weight[weight_offset+int(channel)]
                *first[input_1_offset+int(channel)*input_width+a];
        }
    input_2_adjoint[ulong(sample)*args.stride_2+component] = result;
}
)METAL";

constexpr std::uint64_t channel_group = 64;
constexpr std::uint64_t zero_group = 256;

}  // namespace

struct MetalTensorProductModule::Impl {
    Device device;
    Pipeline zero;
    Pipeline forward;
    Pipeline reverse_channels;
    Pipeline reverse_harmonics;
    std::unique_ptr<MetalStaging> staging;
    MetalTensorProductStatistics statistics;

    TensorProductArgs arguments(
        const MetalTensorProductPlan& plan, const std::size_t samples,
        const std::size_t stride_1, const std::size_t stride_2,
        const std::size_t stride_weights, const std::size_t stride_output) const
    {
        if (samples > std::numeric_limits<std::uint32_t>::max())
            staging->fail("sample count exceeds the 32-bit grid range");
        if (plan.instruction_stride < 7 || plan.sparse_stride < 2
                || plan.multiplicity == 0 || plan.instruction_count == 0)
            staging->fail("tensor product plan is incomplete");
        TensorProductArgs args;
        args.stride_1 = stride_1;
        args.stride_2 = stride_2;
        args.stride_weights = stride_weights;
        args.stride_output = stride_output;
        args.samples = static_cast<std::uint32_t>(samples);
        args.multiplicity = plan.multiplicity;
        args.instruction_count = static_cast<std::uint32_t>(plan.instruction_count);
        args.instruction_stride = static_cast<std::uint32_t>(plan.instruction_stride);
        args.sparse_stride = static_cast<std::uint32_t>(plan.sparse_stride);
        args.harmonic_term_stride = static_cast<std::uint32_t>(plan.harmonic_term_stride);
        args.input_2_dimension = static_cast<std::uint32_t>(plan.input_2_dimension);
        return args;
    }

    const Buffer& table(const void* host, const std::size_t bytes)
    {
        return staging->persistent(host, std::max<std::size_t>(bytes, 4));
    }

    void bind_common_tables(
        CommandBatch& batch, const MetalTensorProductPlan& plan)
    {
        batch.set_buffer(0, table(plan.instruction_data,
                plan.instruction_count*plan.instruction_stride*sizeof(std::int32_t)))
            .set_buffer(1, table(plan.path_weights, plan.instruction_count*sizeof(float)))
            .set_buffer(5, table(plan.output_mask, plan.output_dimension*sizeof(float)));
    }

    void zero_fill(CommandBatch& batch, const DeviceSpan& span, const std::size_t count)
    {
        const std::uint64_t elements = count;
        batch.set_buffer(0, *span.buffer, span.offset)
            .set_value(1, elements)
            .dispatch_threads(zero, {std::max<std::uint64_t>(elements, 1)}, {zero_group});
    }
};

MetalTensorProductModule::MetalTensorProductModule()
    : impl_(std::make_unique<Impl>())
{
    impl_->device = Device::system_default();
    impl_->staging = std::make_unique<MetalStaging>(impl_->device, "tensor product");
    const Library library = impl_->device.compile(kernel_source, module_compile_options());
    impl_->zero = impl_->device.pipeline(library, "symmetrix_tp_zero");
    impl_->forward = impl_->device.pipeline(library, "symmetrix_tp_forward");
    impl_->reverse_channels = impl_->device.pipeline(
        library, "symmetrix_tp_reverse_channels");
    impl_->reverse_harmonics = impl_->device.pipeline(
        library, "symmetrix_tp_reverse_harmonics");
}

MetalTensorProductModule::~MetalTensorProductModule() = default;

const MetalTensorProductStatistics& MetalTensorProductModule::statistics() const
{
    return impl_->statistics;
}

void MetalTensorProductModule::set_host_memory(
    std::shared_ptr<const HostMemoryMap> host_memory)
{
    impl_->staging->set_host_memory(std::move(host_memory));
}

void MetalTensorProductModule::release_persistent_tables()
{
    impl_->staging->release_tables();
}

void MetalTensorProductModule::forward(
    const MetalTensorProductPlan& plan, const std::size_t samples,
    const MetalTensorProductOperand& input_1,
    const MetalTensorProductOperand& input_2,
    const MetalTensorProductOperand& weights,
    const MetalTensorProductResult& output)
{
    auto& m = *impl_;
    if (samples == 0)
        return;
    const auto start = Clock::now();
    const TensorProductArgs args = m.arguments(
        plan, samples, input_1.stride, input_2.stride, weights.stride, output.stride);
    auto batch = m.device.begin();
    m.staging->begin();
    const DeviceSpan first = m.staging->input("input_1", input_1.data, input_1.span);
    const DeviceSpan second = m.staging->input("input_2", input_2.data, input_2.span);
    const DeviceSpan weight = m.staging->input("weights", weights.data, weights.span);
    const DeviceSpan out = m.staging->output("output", output.data, output.span);
    m.zero_fill(batch, out, output.span);
    m.bind_common_tables(batch, plan);
    batch.set_buffer(2, m.table(plan.sparse_indices,
            plan.sparse_count*plan.sparse_stride*sizeof(std::int32_t)))
        .set_buffer(3, m.table(plan.sparse_values, plan.sparse_count*sizeof(float)))
        .set_buffer(4, m.table(plan.component_offsets,
            plan.component_offset_count*sizeof(std::int32_t)))
        .set_buffer(6, *first.buffer, first.offset)
        .set_buffer(7, *second.buffer, second.offset)
        .set_buffer(8, *weight.buffer, weight.offset)
        .set_buffer(9, *out.buffer, out.offset)
        .set_value(10, args)
        .dispatch_threads(m.forward, {plan.multiplicity, samples},
            {std::min<std::uint64_t>(plan.multiplicity, channel_group), 1});
    m.statistics.staging_seconds += seconds_since(start);
    m.statistics.gpu_seconds += batch.submit_and_wait().gpu_seconds;
    m.staging->complete();
    m.statistics.forward_launches += 1;
}

void MetalTensorProductModule::reverse(
    const MetalTensorProductPlan& plan, const std::size_t samples,
    const MetalTensorProductOperand& input_1,
    const MetalTensorProductOperand& input_2,
    const MetalTensorProductOperand& weights,
    const MetalTensorProductOperand& output_adjoint,
    const MetalTensorProductResult& input_1_adjoint,
    const MetalTensorProductResult& input_2_adjoint,
    const MetalTensorProductResult& weights_adjoint)
{
    auto& m = *impl_;
    if (samples == 0)
        return;
    if (input_1_adjoint.stride != input_1.stride
            || input_2_adjoint.stride != input_2.stride
            || weights_adjoint.stride != weights.stride)
        m.staging->fail("adjoint strides differ from their operands");
    const auto start = Clock::now();
    const TensorProductArgs args = m.arguments(plan, samples, input_1.stride,
        input_2.stride, weights.stride, output_adjoint.stride);
    auto batch = m.device.begin();
    m.staging->begin();
    const DeviceSpan first = m.staging->input("input_1", input_1.data, input_1.span);
    const DeviceSpan second = m.staging->input("input_2", input_2.data, input_2.span);
    const DeviceSpan weight = m.staging->input("weights", weights.data, weights.span);
    const DeviceSpan adjoint = m.staging->input(
        "output_adjoint", output_adjoint.data, output_adjoint.span);
    const DeviceSpan first_adjoint = m.staging->output(
        "input_1_adjoint", input_1_adjoint.data, input_1_adjoint.span);
    const DeviceSpan second_adjoint = m.staging->output(
        "input_2_adjoint", input_2_adjoint.data, input_2_adjoint.span);
    const DeviceSpan weight_adjoint = m.staging->output(
        "weights_adjoint", weights_adjoint.data, weights_adjoint.span);
    m.zero_fill(batch, first_adjoint, input_1_adjoint.span);
    m.zero_fill(batch, second_adjoint, input_2_adjoint.span);
    m.zero_fill(batch, weight_adjoint, weights_adjoint.span);

    m.bind_common_tables(batch, plan);
    batch.set_buffer(2, m.table(plan.sparse_indices,
            plan.sparse_count*plan.sparse_stride*sizeof(std::int32_t)))
        .set_buffer(3, m.table(plan.sparse_values, plan.sparse_count*sizeof(float)))
        .set_buffer(4, m.table(plan.component_offsets,
            plan.component_offset_count*sizeof(std::int32_t)))
        .set_buffer(6, *first.buffer, first.offset)
        .set_buffer(7, *second.buffer, second.offset)
        .set_buffer(8, *weight.buffer, weight.offset)
        .set_buffer(9, *adjoint.buffer, adjoint.offset)
        .set_buffer(10, *first_adjoint.buffer, first_adjoint.offset)
        .set_buffer(11, *weight_adjoint.buffer, weight_adjoint.offset)
        .set_value(12, args)
        .dispatch_threads(m.reverse_channels, {plan.multiplicity, samples},
            {std::min<std::uint64_t>(plan.multiplicity, channel_group), 1});

    m.bind_common_tables(batch, plan);
    batch.set_buffer(2, m.table(plan.harmonic_offsets,
            (plan.input_2_dimension+1)*sizeof(std::int32_t)))
        .set_buffer(3, m.table(plan.harmonic_terms,
            plan.harmonic_term_count*plan.harmonic_term_stride*sizeof(std::int32_t)))
        .set_buffer(4, m.table(plan.harmonic_values, plan.harmonic_term_count*sizeof(float)))
        .set_buffer(6, *first.buffer, first.offset)
        .set_buffer(7, *weight.buffer, weight.offset)
        .set_buffer(8, *adjoint.buffer, adjoint.offset)
        .set_buffer(9, *second_adjoint.buffer, second_adjoint.offset)
        .set_value(10, args)
        .dispatch_threads(m.reverse_harmonics, {plan.input_2_dimension, samples},
            {std::min<std::uint64_t>(plan.input_2_dimension, channel_group), 1});
    m.statistics.staging_seconds += seconds_since(start);
    m.statistics.gpu_seconds += batch.submit_and_wait().gpu_seconds;
    m.staging->complete();
    m.statistics.reverse_launches += 1;
}

}  // namespace symmetrix::execution::metal
