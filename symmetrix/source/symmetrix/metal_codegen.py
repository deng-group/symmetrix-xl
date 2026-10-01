"""Render Metal Shading Language kernels for the FP32 R1 interaction.

The kernels reuse the host R1 owner renderers with single-channel tiles, so one
GPU thread owns one (node, channel) output in the forward and source passes and
one edge in the force pass. Metal has no FP64, so the Metal packets carry FP32
radii, cutoff, unit directions, and directed forces; the host converts the
double-precision evaluator views at the packet boundary.

The packet layouts must match ``libsymmetrix/source/metal/metal_r1_abi.hpp``.
"""

from __future__ import annotations

import re
from collections import Counter

from . import jit_codegen as _host
from .execution_contract import normalize_jit_r1_contract

METAL_R1_ABI_VERSION = 1

_PRELUDE = """#include <metal_stdlib>
using namespace metal;

typedef float Scalar;

struct MetalR1RadialSpline {
    uint edge_types;
    uint intervals;
    uint functions;
    float h;
    float x0;
    uint reserved;
    device const float* coefficients;
};

struct MetalR1ForwardArgs {
    uint active_type_count;
    uint num_nodes;
    float cutoff;
    uint reserved;
    device const int* node_types;
    device const int* num_neigh;
    device const int* first_neigh;
    device const int* neigh_indices;
    device const int* neigh_types;
    device const int* type_to_active;
    device const float* radius;
    MetalR1RadialSpline radial;
    device const float* harmonics_values;
    device const float* neighbor_features;
    device float* output;
};

struct MetalR1SourceArgs {
    uint active_type_count;
    uint num_nodes;
    float cutoff;
    uint reserved;
    device const int* node_types;
    device const int* neigh_types;
    device const int* source_offsets;
    device const int* source_edges;
    device const int* edge_receivers;
    device const int* type_to_active;
    device const float* radius;
    MetalR1RadialSpline radial;
    device const float* harmonics_values;
    device const float* output_adjoint;
    device float* source_adjoint;
};

struct MetalR1EdgeArgs {
    uint active_type_count;
    uint num_edges;
    float cutoff;
    uint has_gradients;
    device const int* node_types;
    device const int* neigh_indices;
    device const int* neigh_types;
    device const int* edge_receivers;
    device const int* type_to_active;
    device const float* unit_xyz;
    device const float* radius;
    MetalR1RadialSpline radial;
    device const float* harmonics_values;
    device const float* harmonics_gradients;
    device const float* neighbor_features;
    device const float* output_adjoint;
    device float* directed_forces;
};

struct EvaluationPoint {
    int interval;
    float x;
    float xx;
    float xxx;
};

inline bool edge_is_active(float cutoff, float radius)
{
    return radius < cutoff;
}

inline EvaluationPoint evaluation_point(
    constant MetalR1RadialSpline& radial, float radius)
{
    int interval = int(floor((radius - radial.x0) / radial.h));
    float x = radius - radial.x0 - radial.h * float(interval);
    if (interval < 0) {
        interval = 0;
        x = 0.0f;
    } else if (interval >= int(radial.intervals)) {
        interval = int(radial.intervals) - 1;
        x = radial.h;
    }
    const float xx = x * x;
    return {interval, x, xx, xx * x};
}

inline ulong radial_index(
    constant MetalR1RadialSpline& radial,
    int edge_type,
    int interval,
    int coefficient,
    int function)
{
    return (((ulong(edge_type) * radial.intervals + ulong(interval))
        * 4u + ulong(coefficient)) * radial.functions + ulong(function));
}

inline Scalar evaluate_radial(
    constant MetalR1RadialSpline& radial,
    int edge_type,
    thread const EvaluationPoint& point,
    int function)
{
    const ulong base = radial_index(radial, edge_type, point.interval, 0, function);
    const ulong stride = radial.functions;
    const Scalar c0 = radial.coefficients[base];
    const Scalar c1 = radial.coefficients[base + stride];
    const Scalar c2 = radial.coefficients[base + 2 * stride];
    const Scalar c3 = radial.coefficients[base + 3 * stride];
    return c0 + c1 * point.x + c2 * point.xx + c3 * point.xxx;
}

inline void evaluate_radial(
    constant MetalR1RadialSpline& radial,
    int edge_type,
    thread const EvaluationPoint& point,
    int function,
    thread Scalar& value,
    thread Scalar& derivative)
{
    const ulong base = radial_index(radial, edge_type, point.interval, 0, function);
    const ulong stride = radial.functions;
    const Scalar c0 = radial.coefficients[base];
    const Scalar c1 = radial.coefficients[base + stride];
    const Scalar c2 = radial.coefficients[base + 2 * stride];
    const Scalar c3 = radial.coefficients[base + 3 * stride];
    value = c0 + c1 * point.x + c2 * point.xx + c3 * point.xxx;
    derivative = c1 + c2 * (2.0f * point.x) + c3 * (3.0f * point.xx);
}

inline int pair_type(int left, int right, int type_count)
{
    return left <= right
        ? left * (2 * type_count - left - 1) / 2 + right
        : right * (2 * type_count - right - 1) / 2 + left;
}

template <typename T>
inline void prefetch_read(T) {}
"""

_EDGE_GEOMETRY = re.compile(
    r"    const float x_over_r = .*?        gradients \+= gradient_offset;\n    \}\n",
    re.DOTALL,
)


def _edge_geometry(edge_harmonics: int) -> str:
    count = 3 * edge_harmonics
    return f"""    const float x_over_r = args->unit_xyz[coordinate_offset];
    const float y_over_r = args->unit_xyz[coordinate_offset + 1];
    const float z_over_r = args->unit_xyz[coordinate_offset + 2];
    float gradients[{count}];
    if (args->has_gradients != 0) {{
        for (int g = 0; g < {count}; ++g)
            gradients[g] = args->harmonics_gradients[gradient_offset + g];
    }} else {{
        direct_harmonic_gradients(
            x_over_r, y_over_r, z_over_r, args->radius[edge], gradients);
    }}
"""


_SIMD_WIDTH = 32


def _simdgroup_edge_owner(source: str, channels: int) -> str:
    """Split one edge's channel loop across the lanes of a SIMD group.

    Lanes read consecutive channels, so feature, adjoint, and spline loads
    coalesce; the force components are reduced with ``simd_sum``. Every lane of
    a SIMD group shares one edge, so the early return for inactive edges is
    uniform across the group.
    """

    edits = (
        (
            "void r1_edge_owner(\n    constant MetalR1EdgeArgs* args,\n    int edge)",
            "void r1_edge_owner(\n    constant MetalR1EdgeArgs* args,\n    int edge,\n"
            "    uint lane)",
        ),
        (
            f"    for (int channel = 0; channel < {channels}; ++channel) {{",
            f"    for (int channel = int(lane); channel < {channels}; "
            f"channel += {_SIMD_WIDTH}) {{",
        ),
        (
            "    args->directed_forces[coordinate_offset] -= "
            "static_cast<float>(local_force_x);\n"
            "    args->directed_forces[coordinate_offset + 1] -= "
            "static_cast<float>(local_force_y);\n"
            "    args->directed_forces[coordinate_offset + 2] -= "
            "static_cast<float>(local_force_z);",
            "    local_force_x = simd_sum(local_force_x);\n"
            "    local_force_y = simd_sum(local_force_y);\n"
            "    local_force_z = simd_sum(local_force_z);\n"
            "    if (lane == 0) {\n"
            "        args->directed_forces[coordinate_offset] -= local_force_x;\n"
            "        args->directed_forces[coordinate_offset + 1] -= local_force_y;\n"
            "        args->directed_forces[coordinate_offset + 2] -= local_force_z;\n"
            "    }",
        ),
    )
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"unexpected R1 edge-owner structure near {old[:40]!r}")
        source = source.replace(old, new)
    return source


def _to_msl(source: str) -> str:
    replacements = (
        (
            "const SymmetrixJitHostR1ForwardArgsV1* args",
            "constant MetalR1ForwardArgs* args",
        ),
        (
            "const SymmetrixJitHostR1SourceArgsV1* args",
            "constant MetalR1SourceArgs* args",
        ),
        ("const SymmetrixJitHostR1EdgeArgsV1* args", "constant MetalR1EdgeArgs* args"),
        ("static_cast<double>(", "static_cast<float>("),
        # MSL 3.0 is C++14-based; the condition is a template constant.
        ("if constexpr (", "if ("),
        ("std::int32_t", "int"),
        ("std::uint32_t", "uint"),
        ("std::size_t", "ulong"),
    )
    for old, new in replacements:
        source = source.replace(old, new)
    source = re.sub(r"\bstd::(sqrt|floor|fabs|abs|fma|min|max)\b", r"\1", source)
    leftovers = sorted(set(re.findall(r"\bstd::\w+|\bdouble\b|\bnullptr\b", source)))
    if leftovers:
        raise ValueError(
            f"R1 source has constructs without an MSL mapping: {leftovers}"
        )
    return source


def _direct_gradient_helper(edge_harmonics: int) -> str:
    helper = _host._render_direct_harmonic_gradient_helper("inline", edge_harmonics)
    marker = "    const Scalar inverse_norm"
    body = helper[helper.index(marker) :]
    header = (
        "inline void direct_harmonic_gradients(\n"
        "    Scalar ux, Scalar uy, Scalar uz, float radius,\n"
        "    thread Scalar* gradients)\n{\n"
    )
    return _to_msl(header + body)


def metal_r1_metadata(contract: dict) -> dict:
    normalized = normalize_jit_r1_contract(contract)
    host = _host.jit_r1_host_plugin_metadata(normalized, None, precision="float32")
    return {
        "abi_version": METAL_R1_ABI_VERSION,
        "channels": host["channels"],
        "edge_harmonics": (host["edge_l_max"] + 1) ** 2,
        "source_harmonics": (host["source_l_max"] + 1) ** 2,
        "output_components": host["output_components"],
        "contract_fingerprint": host["contract_fingerprint"],
    }


def render_jit_r1_metal_source(contract: dict) -> str:
    """Render the FP32 Metal R1 forward, source, and edge kernels."""

    normalized = normalize_jit_r1_contract(contract)
    metadata = metal_r1_metadata(normalized)
    path_rows = _host._fixed_weight_path_rows(normalized)
    channels = metadata["channels"]
    edge_harmonics = metadata["edge_harmonics"]
    source_harmonics = metadata["source_harmonics"]
    output_components = metadata["output_components"]

    forward = _host._render_tiled_host_forward_owner(
        path_rows,
        channels=channels,
        edge_harmonics=edge_harmonics,
        source_harmonics=source_harmonics,
        output_components=output_components,
        channel_tile=1,
    )
    source = _host._render_tiled_host_source_owner(
        path_rows,
        channels=channels,
        edge_harmonics=edge_harmonics,
        source_harmonics=source_harmonics,
        output_components=output_components,
        channel_tile=1,
    )
    edge, replaced = _EDGE_GEOMETRY.subn(
        _edge_geometry(edge_harmonics),
        _host._render_edge_owner(
            path_rows,
            channels=channels,
            edge_harmonics=edge_harmonics,
            source_harmonics=source_harmonics,
        ),
    )
    if replaced != 1:
        raise ValueError("unexpected R1 edge-owner geometry preamble")
    loaders = _host._render_row_loaders(path_rows, channels, output_components)

    kernels = f"""
kernel void symmetrix_r1_forward(
    constant MetalR1ForwardArgs& args [[buffer(0)]],
    uint flat [[thread_position_in_grid]])
{{
    const uint receiver = flat / {channels}u;
    if (receiver >= args.num_nodes)
        return;
    r1_forward_owner(&args, int(receiver), int(flat % {channels}u));
}}

kernel void symmetrix_r1_source(
    constant MetalR1SourceArgs& args [[buffer(0)]],
    uint flat [[thread_position_in_grid]])
{{
    const uint source = flat / {channels}u;
    if (source >= args.num_nodes)
        return;
    r1_source_owner_impl<true>(&args, int(source), int(flat % {channels}u));
}}

kernel void symmetrix_r1_edge(
    constant MetalR1EdgeArgs& args [[buffer(0)]],
    uint flat [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const uint edge = flat / {_SIMD_WIDTH}u;
    if (edge >= args.num_edges)
        return;
    r1_edge_owner(&args, int(edge), lane);
}}
"""
    return "\n".join(
        (
            f"// Generated by symmetrix.metal_codegen (Metal R1 ABI "
            f"{METAL_R1_ABI_VERSION}). Do not edit.",
            f"// contract: {metadata['contract_fingerprint']}",
            _PRELUDE,
            _direct_gradient_helper(edge_harmonics),
            _to_msl(loaders),
            _to_msl(forward),
            _to_msl(source),
            _simdgroup_edge_owner(_to_msl(edge), channels),
            kernels,
        )
    )


METAL_M0_ABI_VERSION = 1


def metal_m0_metadata(contract: dict) -> dict:
    from .jit_operator_codegen import jit_m0_host_metadata

    host = jit_m0_host_metadata(contract, precision="float32")
    normalized = host["contract"]
    return {
        "abi_version": METAL_M0_ABI_VERSION,
        "channels": int(normalized["channels"]),
        "input_components": int(host["input_components"]),
        "output_components": int(host["output_components"]),
        "term_count": int(host["term_count"]),
        "structure_fingerprint": host["structure_fingerprint"],
    }


def render_jit_m0_metal_source(contract: dict) -> str:
    """Render FP32 Metal M0 forward and reverse kernels from the term list.

    One thread owns one (node, channel) pair. Channels are padded to whole
    SIMD groups so a group never spans two nodes; the reverse pass reduces the
    input-scale adjoint over each group into one FP32 partial for the host to
    accumulate in FP64.
    """

    from .jit_operator_codegen import _m0_terms, jit_m0_host_metadata

    metadata = metal_m0_metadata(contract)
    terms = _m0_terms(jit_m0_host_metadata(contract, precision="float32")["contract"])
    if len(terms) != metadata["term_count"]:
        raise ValueError("M0 term list does not match the contract metadata")
    forward = []
    reverse = []
    for term, (output, components) in enumerate(terms):
        product = "*".join(f"x[{component}]" for component in components)
        forward.append(f"    out[{output}] += weights[{term}*C]*{product};")
        for component, multiplicity in Counter(components).items():
            remaining = list(components)
            remaining.remove(component)
            factor = (
                "*".join(f"x[{value}]" for value in remaining) if remaining else "1.0f"
            )
            if multiplicity != 1:
                factor = f"({multiplicity}.0f*{factor})"
            reverse.append(
                f"        grad[{component}] += adj[{output}]*weights[{term}*C]*{factor};"
            )
    return f"""// Generated by symmetrix.metal_codegen (Metal M0 ABI {METAL_M0_ABI_VERSION}). Do not edit.
// structure: {metadata["structure_fingerprint"]}
#include <metal_stdlib>
using namespace metal;

constant constexpr uint C = {metadata["channels"]}u;
constant constexpr uint IN = {metadata["input_components"]}u;
constant constexpr uint OUT = {metadata["output_components"]}u;
constant constexpr uint TERMS = {metadata["term_count"]}u;

struct MetalM0Args {{
    uint num_nodes;
    uint capture_input_scale_adjoint;
    uint padded_channels;
    uint reserved;
    device const int* node_types;
    device const float* input;
    device const float* weights;
    device const float* output_adjoint;
    device float* output;
    device float* input_adjoint;
    device float* scale_partials;
}};

kernel void symmetrix_m0_forward(
    constant MetalM0Args& a [[buffer(0)]],
    uint flat [[thread_position_in_grid]])
{{
    const uint node = flat / a.padded_channels;
    const uint channel = flat % a.padded_channels;
    if (node >= a.num_nodes || channel >= C)
        return;
    float x[IN];
    for (uint component = 0; component < IN; ++component)
        x[component] = a.input[(ulong(node) * IN + component) * C + channel];
    device const float* weights =
        a.weights + ulong(a.node_types[node]) * TERMS * C + channel;
    float out[OUT];
    for (uint component = 0; component < OUT; ++component)
        out[component] = 0.0f;
{chr(10).join(forward)}
    for (uint component = 0; component < OUT; ++component)
        a.output[(ulong(node) * OUT + component) * C + channel] = out[component];
}}

kernel void symmetrix_m0_reverse(
    constant MetalM0Args& a [[buffer(0)]],
    uint flat [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const uint node = flat / a.padded_channels;
    if (node >= a.num_nodes)
        return;
    const uint channel = flat % a.padded_channels;
    float scale = 0.0f;
    if (channel < C) {{
        float x[IN];
        float grad[IN];
        float adj[OUT];
        for (uint component = 0; component < IN; ++component) {{
            x[component] = a.input[(ulong(node) * IN + component) * C + channel];
            grad[component] = 0.0f;
        }}
        for (uint component = 0; component < OUT; ++component)
            adj[component] =
                a.output_adjoint[(ulong(node) * OUT + component) * C + channel];
        device const float* weights =
            a.weights + ulong(a.node_types[node]) * TERMS * C + channel;
{chr(10).join(reverse)}
        for (uint component = 0; component < IN; ++component) {{
            a.input_adjoint[(ulong(node) * IN + component) * C + channel] =
                grad[component];
            scale += x[component] * grad[component];
        }}
    }}
    if (a.capture_input_scale_adjoint != 0) {{
        scale = simd_sum(scale);
        if (lane == 0)
            a.scale_partials[ulong(node) * (a.padded_channels / 32u)
                + channel / 32u] = scale;
    }}
}}
"""
