"""Thin ASE-facing adapter for the complete native MACE Polar family.

The adapter owns only compact input validation/staging, native topology and
geometry preparation, one whole-model invocation, and public-result
translation.  All scientific composition and reverse execution remains in
``MacePolarKokkosT``.
"""

from __future__ import annotations

import numpy as np

from . import family_registry


class MacePolarFrontendAdapter:
    """Family-owned bridge from compact ASE data to the native evaluator."""

    _DIAGNOSTIC_KEYS = (
        "requested_solver",
        "resolved_solver",
        "solver_reason",
        "automatic_atom_count",
        "automatic_reciprocal_mode_count",
        "automatic_reciprocal_mode_count_is_lower_bound",
        "automatic_atom_mode_pair_count",
        "automatic_atom_mode_pair_threshold",
        "n_k",
        "reciprocal_allocation_elements",
        "reciprocal_allocation_bytes",
        "reciprocal_table_elements",
        "reciprocal_table_bytes",
        "particle_mesh_active_bytes",
        "particle_mesh_capacity_bytes",
        "particle_mesh_volume",
        "particle_mesh_shape",
        "particle_mesh_rebuilds",
        "particle_mesh_reuses",
        "particle_mesh_forward_plan_builds",
        "particle_mesh_forward_plan_reuses",
        "particle_mesh_inverse_plan_builds",
        "particle_mesh_inverse_plan_reuses",
        "backbone_fallback_count",
        "polar_fallback_count",
        "fallback_count",
        "reverse_strategy",
        "cumulative_transaction_count",
        "cumulative_polar_reverse_count",
        "backbone_evaluation_count",
        "backbone_forward_layer_count",
        "backbone_reverse_layer_count",
        "seeded_backbone_evaluation_count",
        "polar_reverse_count",
        "reverse_executed",
        "product_seed_cleared",
    )

    @staticmethod
    def _program_has_dipole(evaluator) -> bool:
        program = getattr(evaluator, "family_program", None)
        return isinstance(program, dict) and any(
            isinstance(node, dict) and node.get("kind") == "dipole_readout"
            for node in program.get("operators", ())
        )

    @classmethod
    def configure(cls, calculator) -> None:
        """Validate the family contract and configure its retained backbone tap."""

        if not cls._program_has_dipole(calculator.evaluator):
            return
        contract = getattr(calculator.evaluator, "polar_long_range_contract", None)
        if not contract:
            raise RuntimeError(
                "the MACE Polar program declares long-range nodes but its "
                "evaluator carries no POLAR_BACKBONE long-range contract"
            )
        shapes = dict(calculator.evaluator.polar_delta_shapes())
        try:
            calculator.evaluator.layer_feature_components(
                int(shapes["node_feature_dimension"])
            )
        except RuntimeError as error:
            raise RuntimeError(
                "the MACE Polar family does not serve this checkpoint's "
                f"node-feature basis: {error}"
            ) from error
        calculator.evaluator.set_retain_layer_features(True)
        calculator._polar_contract = dict(contract)
        calculator._polar_shapes = shapes
        if "dipole" not in calculator.implemented_properties:
            calculator.implemented_properties.append("dipole")
        if "energies" in calculator.implemented_properties:
            calculator.implemented_properties.remove("energies")

    @staticmethod
    def _resolve_scalar(name, info, explicit, default) -> float:
        if explicit is not None:
            value = explicit
        elif name in info:
            value = info[name]
        else:
            return default
        if isinstance(value, bool) or not np.isscalar(value):
            raise ValueError(f"the MACE Polar {name} input must be a scalar")
        resolved = float(value)
        if not np.isfinite(resolved):
            raise ValueError(f"the MACE Polar {name} input must be finite")
        return resolved

    @classmethod
    def resolve_graph_inputs(cls, calculator, atoms) -> dict[str, object]:
        """Resolve explicit, structure-info, then neutral graph inputs."""

        info = getattr(atoms, "info", None) or {}
        return {
            "total_charge": cls._resolve_scalar(
                "total_charge", info, calculator._total_charge, 0.0
            ),
            "total_spin": cls._resolve_scalar(
                "total_spin", info, calculator._total_spin, 0.0
            ),
            "fermi_level": cls._resolve_scalar(
                "fermi_level", info, calculator._fermi_level, 0.0
            ),
            "electric_field": calculator._resolve_electric_field(atoms),
        }

    @classmethod
    def inputs_unchanged(cls, calculator, atoms, tol: float) -> bool:
        previous = calculator._polar_graph_inputs
        current = cls.resolve_graph_inputs(calculator, atoms)
        if previous is None:
            return False
        for key in ("total_charge", "total_spin", "fermi_level"):
            if not np.isclose(previous[key], current[key], rtol=0.0, atol=tol):
                return False
        return bool(
            np.allclose(
                previous["electric_field"],
                current["electric_field"],
                rtol=0.0,
                atol=tol,
            )
        )

    @staticmethod
    def _prepare_graph_geometry(calculator, atoms, mace_inputs) -> int:
        """Prepare native topology and Cartesian geometry without evaluation."""

        family = calculator._resolved_family_entry()
        cache = calculator._neighbor_cache
        if cache is None:
            raise RuntimeError("MACE Polar requires a prepared neighbor cache")
        if cache.device_graph_generation != 0:
            if cache.device_geometry_generation != cache.geometry_generation:
                device_builder = getattr(
                    calculator.evaluator,
                    "_prepare_periodic_factorized_graph",
                    None,
                )
                if not callable(device_builder):
                    raise family_registry.MissingCapabilityError(
                        "the MACE Polar evaluator cannot refresh its native "
                        "periodic graph after a cell-geometry change"
                    )
                device_graph = device_builder(
                    np.asarray(cache.node_types, dtype=np.int32),
                    np.ascontiguousarray(
                        atoms.get_positions(), dtype=np.float64
                    ).reshape(-1),
                    np.ascontiguousarray(cache.cell, dtype=np.float64).reshape(-1),
                    np.ascontiguousarray(cache.inverse_cell, dtype=np.float64).reshape(
                        -1
                    ),
                    calculator.cutoff + calculator.neighbor_skin,
                )
                calculator._sync_low_memory_policy_state()
                cache.device_graph_generation = int(device_graph["generation"])
                cache.device_geometry_generation = cache.geometry_generation
                cache.num_edges = int(device_graph["num_edges"])
            return int(cache.device_graph_generation)
        (prepare_graph,) = family_registry.require_surface(
            family, calculator.evaluator, "prepared_graph"
        )
        (prepare_geometry,) = family_registry.require_surface(
            family, calculator.evaluator, "prepared_geometry"
        )
        num_nodes, node_types, num_neigh, j_list, neigh_types, _, _, _ = mace_inputs
        generation = prepare_graph(
            num_nodes,
            np.asarray(node_types, dtype=np.int32),
            np.asarray(num_neigh, dtype=np.int32),
            np.asarray(j_list, dtype=np.int32),
            np.asarray(neigh_types, dtype=np.int32),
        )
        calculator._sync_low_memory_policy_state()
        cache = calculator._materialize_neighbor_cache_geometry()
        # A materialized reference graph contains only active edges, whereas
        # the Verlet cache also owns its skin shell. Bind the active vectors
        # to their current positions rather than mixing the two topologies.
        materialized = calculator.evaluator.streamed_edges_mode == "materialized"
        reference_positions = (
            atoms.get_positions() if materialized else cache.reference_positions
        )
        reference_xyz = mace_inputs[5] if materialized else cache.reference_xyz
        prepare_geometry(
            generation,
            np.ascontiguousarray(reference_positions, dtype=np.float64).reshape(-1),
            np.ascontiguousarray(reference_xyz, dtype=np.float64).reshape(-1),
            np.ascontiguousarray(cache.cell, dtype=np.float64).reshape(-1),
            np.ascontiguousarray(cache.inverse_cell, dtype=np.float64).reshape(-1),
            np.ascontiguousarray(cache.pbc, dtype=np.int32).reshape(-1),
        )
        return int(generation)

    @classmethod
    def evaluate(cls, calculator, atoms, properties, mace_inputs) -> dict[str, object]:
        """Run exactly one native whole-model transaction and return ASE outputs."""

        family = calculator._resolved_family_entry()
        (evaluate_prepared,) = family_registry.require_surface(
            family, calculator.evaluator, "whole_model"
        )
        generation = cls._prepare_graph_geometry(calculator, atoms, mace_inputs)
        graph_inputs = cls.resolve_graph_inputs(calculator, atoms)
        electric_field = np.asarray(
            graph_inputs["electric_field"], dtype=np.float64
        ).reshape(3)
        requested = set(properties)
        native_result = evaluate_prepared(
            graph_generation=generation,
            node_types=np.asarray(mace_inputs[1], dtype=np.int32),
            positions=np.ascontiguousarray(atoms.get_positions(), dtype=np.float64),
            cell=np.ascontiguousarray(np.asarray(atoms.cell, dtype=np.float64)).reshape(
                1, 3, 3
            ),
            graph_pbc=np.asarray(atoms.pbc, dtype=np.int32).reshape(1, 3),
            node_graph_index=np.zeros(len(atoms), dtype=np.int32),
            total_charge=[float(graph_inputs["total_charge"])],
            total_spin=[float(graph_inputs["total_spin"])],
            external_potential=np.asarray([[0.0, *electric_field]], dtype=np.float64),
            include_forces=bool({"forces", "stress"} & requested),
            include_stress="stress" in requested,
            include_dipole="dipole" in requested,
            polar_electrostatics=calculator.polar_electrostatics,
            pme_mesh=calculator.polar_pme_mesh,
            pme_spline_order=calculator.polar_pme_order,
        )
        calculator._polar_graph_inputs = graph_inputs
        calculator.polar_electrostatics_diagnostics = {
            key: native_result[key]
            for key in cls._DIAGNOSTIC_KEYS
            if key in native_result
        }
        results: dict[str, object] = {}
        if "energy" in requested:
            results["energy"] = float(native_result["energy"])
            results["free_energy"] = float(native_result["free_energy"])
        elif "free_energy" in requested:
            results["free_energy"] = float(native_result["free_energy"])
        if "forces" in requested:
            results["forces"] = np.asarray(native_result["forces"], dtype=float)
        if "stress" in requested:
            results["stress"] = np.asarray(
                native_result["stress"], dtype=float
            ).reshape(6)
        if "dipole" in requested:
            results["dipole"] = np.asarray(
                native_result["dipole"], dtype=float
            ).reshape(3)
        return results

    @staticmethod
    def _batch_topology(calculator, atoms):
        """Return the neighbor topology of one batch member with lattice shifts.

        Only edges inside the cutoff at the current positions are returned.
        A single evaluation also passes the neighbor-skin shell, clamped to
        the cutoff, where the radial functions of trained models vanish. For a
        trajectory with an atombatch source, the calculator's Verlet cache
        supplies the topology, which is rebuilt only when atoms move beyond the
        neighbor skin; otherwise the neighbor list is built for this call and
        the shifts are recovered from the edge vectors.
        """

        prepared = (
            getattr(calculator.evaluator, "streamed_edges_mode", None) == "direct"
        )
        if prepared and atoms.info.get("atombatch_source") is not None:
            calculator._mace_inputs(
                atoms, native_geometry=True, allow_device_neighbor_graph=False
            )
            cache = calculator._neighbor_cache
            if cache is not None and cache.device_graph_generation == 0:
                # The Verlet topology includes a skin shell beyond the cutoff;
                # only the edges inside the cutoff at the current positions are
                # passed, which are the edges of an independent evaluation.
                receivers = np.asarray(cache.receivers, dtype=np.int64)
                sources = np.asarray(cache.sources, dtype=np.int64)
                shifts = np.asarray(cache.shifts, dtype=np.int32).reshape(-1, 3)
                positions = np.asarray(atoms.get_positions(), dtype=np.float64)
                vectors = positions[sources] - positions[receivers]
                if atoms.pbc.any():
                    vectors += shifts @ np.asarray(atoms.cell.array, dtype=np.float64)
                inside = np.einsum("ij,ij->i", vectors, vectors) < calculator.cutoff**2
                return (
                    len(atoms),
                    np.asarray(cache.node_types, dtype=np.int32),
                    np.bincount(receivers[inside], minlength=len(atoms)).astype(
                        np.int32
                    ),
                    sources[inside],
                    np.asarray(cache.neigh_types, dtype=np.int32)[inside],
                    shifts[inside],
                )
        num_nodes, types, _, j, types_j, xyz, _, i = calculator._mace_inputs(atoms)
        j = np.asarray(j, dtype=np.int64)
        i = np.asarray(i, dtype=np.int64)
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        if atoms.pbc.any():
            cell = np.asarray(atoms.cell.array, dtype=np.float64)
            translation = xyz - (positions[j] - positions[i])
            shift = np.rint(translation @ np.linalg.inv(cell)).astype(np.int32)
            vectors = positions[j] - positions[i] + shift @ cell
        else:
            shift = np.zeros((len(j), 3), dtype=np.int32)
            vectors = positions[j] - positions[i]
        inside = np.einsum("ij,ij->i", vectors, vectors) < calculator.cutoff**2
        return (
            num_nodes,
            np.asarray(types, dtype=np.int32),
            np.bincount(i[inside], minlength=num_nodes).astype(np.int32),
            j[inside],
            np.asarray(types_j, dtype=np.int32)[inside],
            shift[inside],
        )

    @classmethod
    def evaluate_batch(cls, calculator, structures, properties) -> list[dict]:
        """Evaluate independent structures in one native transaction.

        The structures form one joined graph whose segment geometry keeps each
        structure in its own cell, so no edge joins two structures, and the
        long-range program treats every structure as its own graph. All
        structures must share one PBC pattern; the native call raises
        ``ValueError`` when they would resolve different long-range solvers or
        the particle-mesh solver. Returns one result dictionary per structure
        with the properties of :meth:`evaluate`.
        """

        evaluator = calculator.evaluator
        requested = set(properties)
        node_types, num_neigh, j_list, neigh_types, shifts = [], [], [], [], []
        positions, cells, segment_cells, pbcs, node_graphs = [], [], [], [], []
        charge, spin, external, edge_offsets = [], [], [], [0]
        offset = 0
        for graph, atoms in enumerate(structures):
            num_nodes, types, neighbors, j, types_j, shift = cls._batch_topology(
                calculator, atoms
            )
            atom_positions = np.asarray(atoms.get_positions(), dtype=np.float64)
            cell = np.asarray(atoms.cell.array, dtype=np.float64)
            segment_cell = cell if atoms.pbc.any() else np.eye(3)
            inputs = cls.resolve_graph_inputs(calculator, atoms)
            field = np.asarray(inputs["electric_field"], dtype=np.float64).reshape(3)
            node_types.append(np.asarray(types, dtype=np.int32))
            num_neigh.append(np.asarray(neighbors, dtype=np.int32))
            j_list.append((np.asarray(j, dtype=np.int64) + offset).astype(np.int32))
            neigh_types.append(np.asarray(types_j, dtype=np.int32))
            shifts.append(shift)
            positions.append(atom_positions)
            cells.append(cell.reshape(-1))
            segment_cells.append(segment_cell.reshape(-1))
            pbcs.append(np.asarray(atoms.pbc, dtype=np.int32))
            node_graphs.append(np.full(num_nodes, graph, dtype=np.int32))
            charge.append(float(inputs["total_charge"]))
            spin.append(float(inputs["total_spin"]))
            external.append((0.0, *field))
            edge_offsets.append(edge_offsets[-1] + len(shift))
            offset += num_nodes

        all_types = np.concatenate(node_types)
        all_positions = np.ascontiguousarray(np.concatenate(positions).reshape(-1))
        offsets = np.asarray(edge_offsets, dtype=np.int64)
        generation = evaluator._prepare_factorized_graph(
            offset,
            all_types,
            np.concatenate(num_neigh),
            np.concatenate(j_list),
            np.concatenate(neigh_types),
        )
        evaluator._prepare_factorized_batch(generation, offsets)
        evaluator._prepare_factorized_segment_geometry(
            generation,
            all_positions,
            np.ascontiguousarray(np.concatenate(shifts).reshape(-1)),
            np.concatenate(segment_cells),
        )
        native = evaluator._evaluate_polar_prepared_batch(
            generation,
            all_types,
            all_positions,
            np.concatenate(cells),
            np.concatenate(pbcs),
            np.concatenate(node_graphs),
            np.asarray(charge, dtype=np.float64),
            np.asarray(spin, dtype=np.float64),
            np.asarray(external, dtype=np.float64).reshape(-1),
            offsets,
            include_forces=bool({"forces", "stress"} & requested),
            include_stress="stress" in requested,
            include_dipole="dipole" in requested,
            polar_electrostatics=calculator.polar_electrostatics,
        )
        results = []
        start = 0
        for graph, atoms in enumerate(structures):
            energy = float(native["energies"][graph])
            result = {"energy": energy, "free_energy": energy}
            if "forces" in requested:
                result["forces"] = np.array(
                    native["forces"][start : start + len(atoms)], dtype=float
                )
            if "stress" in requested:
                result["stress"] = np.array(native["stress"][graph], dtype=float)
            if "dipole" in requested:
                result["dipole"] = np.array(native["dipoles"][graph], dtype=float)
            results.append(result)
            start += len(atoms)
        return results
