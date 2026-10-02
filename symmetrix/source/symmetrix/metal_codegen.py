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


_EDGE_LOADER = re.compile(
    r"inline float load_row_(\d+)\(\n    constant MetalR1EdgeArgs\* args,\n"
    r"    int receiver,\n    int channel\)\n\{\n(.*?)\n\}",
    re.DOTALL,
)
_ADJOINT_READ = re.compile(
    r"args->output_adjoint\[\(static_cast<ulong>\(receiver\) \* \d+ \+ (\d+)\)"
    r" \* (\d+) \+ channel\]"
)

# Channel block of the blocked edge kernel: one SIMD lane per channel, so a
# receiver's adjoint tile is output_components * 32 floats.
METAL_R1_EDGE_BLOCK_CHANNELS = _SIMD_WIDTH


def _blocked_edge_loaders(loaders: str) -> str:
    """Row loaders reading one 32-channel block of a receiver's adjoint tile."""

    rendered = []
    for row, body in _EDGE_LOADER.findall(loaders):
        body, reads = _ADJOINT_READ.subn(
            rf"tile[\1 * {METAL_R1_EDGE_BLOCK_CHANNELS} + local]", body
        )
        if reads == 0 or "args->" in body or "receiver" in body:
            raise ValueError(f"unexpected R1 edge row loader {row}")
        rendered.append(
            f"inline float block_row_{row}(\n"
            "    threadgroup const float* tile,\n"
            "    int local)\n{\n" + body.replace("channel", "local") + "\n}"
        )
    if not rendered:
        raise ValueError("R1 source has no edge row loaders")
    return "\n\n".join(rendered)


def _blocked_edge_owner(edge_owner: str, channels: int) -> str:
    """Edge owner for one 32-channel block; lane l owns channel base + l and
    the block's partial force is written to its own force plane."""

    edits = (
        (
            "void r1_edge_owner(\n    constant MetalR1EdgeArgs* args,\n    int edge,\n"
            "    uint lane)",
            "void r1_edge_owner_blocked(\n    constant MetalR1EdgeArgs* args,\n"
            "    int edge,\n    uint lane,\n    int block_base,\n"
            "    threadgroup const float* tile,\n    device float* forces)",
        ),
        (
            f"    for (int channel = int(lane); channel < {channels}; "
            f"channel += {_SIMD_WIDTH}) {{",
            "    {\n        const int channel = block_base + int(lane);",
        ),
    )
    owner = edge_owner
    for old, new in edits:
        if owner.count(old) != 1:
            raise ValueError(f"unexpected R1 edge-owner structure near {old[:40]!r}")
        owner = owner.replace(old, new)
    owner, loads = re.subn(
        r"load_row_(\d+)\(args, receiver, channel\)",
        r"block_row_\1(tile, int(lane))",
        owner,
    )
    owner, writes = re.subn(r"args->directed_forces\[", "forces[", owner)
    if loads == 0 or writes != 3:
        raise ValueError("unexpected R1 edge-owner structure for channel blocks")
    return owner


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
    loaders_msl = _to_msl(loaders)
    edge_msl = _simdgroup_edge_owner(_to_msl(edge), channels)
    tile_floats = output_components * channels

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
    block = METAL_R1_EDGE_BLOCK_CHANNELS
    blocked = channels % block == 0
    if blocked:
        block_floats = output_components * block
        kernels += f"""
// One threadgroup per (receiver, 32-channel block) of a receiver-ordered edge
// list, flattened receiver-major. The block's adjoint rows are staged in threadgroup memory, small
// enough for several threadgroups per core, and each block writes a partial
// force plane that the host sums in a fixed order.
kernel void symmetrix_r1_edge_blocked(
    constant MetalR1EdgeArgs& args [[buffer(0)]],
    device const int* receiver_offsets [[buffer(1)]],
    device float* block_forces [[buffer(2)]],
    uint position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint group [[simdgroup_index_in_threadgroup]],
    uint groups [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    threadgroup float tile[{block_floats}];
    const uint receiver = position / {channels // block}u;
    const uint block_index = position % {channels // block}u;
    const uint block_base = block_index * {block}u;
    const int begin = receiver_offsets[receiver];
    const int end = receiver_offsets[receiver + 1];
    if (begin == end)
        return;
    device const float* rows =
        args.output_adjoint + ulong(receiver) * {tile_floats}u + block_base;
    for (uint index = thread_index; index < {block_floats}u; index += threads)
        tile[index] = rows[(index / {block}u) * {channels}u + index % {block}u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* forces = block_forces + ulong(block_index) * 3u * args.num_edges;
    for (int edge = begin + int(group); edge < end; edge += int(groups))
        r1_edge_owner_blocked(&args, edge, lane, int(block_base), tile, forces);
}}
"""
    return "\n".join(
        (
            f"// Generated by symmetrix.metal_codegen (Metal R1 ABI "
            f"{METAL_R1_ABI_VERSION}). Do not edit.",
            f"// contract: {metadata['contract_fingerprint']}",
            _PRELUDE,
            _direct_gradient_helper(edge_harmonics),
            loaders_msl,
            _to_msl(forward),
            _to_msl(source),
            edge_msl,
            *(
                (
                    _blocked_edge_loaders(loaders_msl),
                    _blocked_edge_owner(edge_msl, channels),
                )
                if blocked
                else ()
            ),
            kernels,
        )
    )


METAL_M0_ABI_VERSION = 1


def _product_terms(contract: dict, output_components: int | None):
    from .jit_operator_codegen import _m0_terms, jit_m0_host_metadata

    host = jit_m0_host_metadata(contract, precision="float32")
    terms = _m0_terms(host["contract"])
    if len(terms) != int(host["term_count"]):
        raise ValueError("M0 term list does not match the contract metadata")
    outputs = int(host["output_components"])
    if output_components is not None:
        if not 0 < output_components <= outputs:
            raise ValueError("requested output components exceed the contract")
        # Terms are grouped by output component, so a prefix keeps the
        # weight index of every retained term.
        kept = [term for term in terms if term[0] < output_components]
        if kept != terms[: len(kept)]:
            raise ValueError("M0 terms are not grouped by output component")
        terms, outputs = kept, output_components
    return host, terms, outputs


def metal_m0_metadata(contract: dict, output_components: int | None = None) -> dict:
    host, terms, outputs = _product_terms(contract, output_components)
    return {
        "abi_version": METAL_M0_ABI_VERSION,
        "channels": int(host["contract"]["channels"]),
        "input_components": int(host["input_components"]),
        "output_components": outputs,
        "term_count": len(terms),
        "structure_fingerprint": host["structure_fingerprint"],
    }


def metal_m1_metadata(m0_contract: dict) -> dict:
    """The standard M1 contraction: the scalar output block of M0's terms."""

    return metal_m0_metadata(m0_contract, output_components=1)


def render_jit_m1_metal_source(m0_contract: dict) -> str:
    return render_jit_m0_metal_source(m0_contract, output_components=1)


def render_jit_m0_metal_source(
    contract: dict, output_components: int | None = None
) -> str:
    """Render FP32 Metal product-basis kernels from the M0 term list.

    One thread owns one (node, channel) pair. Channels are padded to whole
    SIMD groups so a group never spans two nodes; the reverse pass reduces the
    input-scale adjoint over each group into one FP32 partial for the host to
    accumulate in FP64. ``output_components`` keeps a leading block of output
    components, which is how the standard M1 contraction is rendered.
    """

    metadata = metal_m0_metadata(contract, output_components)
    _, terms, _ = _product_terms(contract, output_components)
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
