"""Batched evaluation of independent structures in one native call."""

import hashlib
from collections import OrderedDict

import numpy as np
from ase.calculators.calculator import all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .calculator import _receiver_major_neighbor_arrays, neighbor_list


BATCH_MODES = ("native", "sequential")
BATCH_PROPERTIES = ("energy", "free_energy", "energies", "forces", "stress")


class BatchEvaluator:
    """Evaluate many structures with one Symmetrix calculator.

    With ``batch_mode="native"``, all structures are placed in one disconnected
    neighbor graph and evaluated by a single native call. Message passing runs
    only along graph edges, so the structures do not interact; energies are
    summed per structure from node energies and stresses are reduced per
    structure from the directed pair forces. Neighbor lists are cached by
    positions, cell, and periodicity, so structures that differ only in species
    (for example, orderings of a disordered lattice) share one neighbor list.

    With ``batch_mode="sequential"``, each structure is evaluated by the calculator's
    ASE path in turn. This supports every calculator feature, including
    dispersion corrections and electric-field models, but pays the per-call
    cost once per structure.
    """

    def __init__(self, calculator, batch_mode="native", neighbor_cache_size=64):
        if batch_mode not in BATCH_MODES:
            raise ValueError(
                f"batch_mode must be one of {BATCH_MODES}, not {batch_mode!r}."
            )
        if batch_mode == "native":
            unsupported = _native_unsupported_reason(calculator)
            if unsupported:
                raise ValueError(
                    f"Native batches do not support {unsupported}; "
                    "use batch_mode='sequential'."
                )
        self.calculator = calculator
        self.batch_mode = batch_mode
        self.neighbor_cache_size = int(neighbor_cache_size)
        self._neighbor_cache = OrderedDict()
        atomic_numbers = [int(value) for value in calculator.evaluator.atomic_numbers]
        self._type_lookup = np.full(max(atomic_numbers) + 1, -1, dtype=np.int64)
        self._type_lookup[atomic_numbers] = np.arange(len(atomic_numbers))

    def calculate(self, structures, properties=("energy", "forces")):
        """Return one ASE-style result dictionary per structure.

        Stress is returned in ASE's six-component Voigt order and sign
        convention and requires three-dimensional cells.
        """
        structures = list(structures)
        properties = tuple(dict.fromkeys(properties))
        unsupported = [prop for prop in properties if prop not in BATCH_PROPERTIES]
        if unsupported:
            raise ValueError(f"Unsupported batch property: {unsupported[0]}")
        if not structures:
            return []
        compute_stress = "stress" in properties
        if compute_stress:
            volumes = np.array([abs(atoms.cell.volume) for atoms in structures])
            if not np.all(volumes > 0.0):
                raise ValueError(
                    "Stress requires three-dimensional cells; omit 'stress' for "
                    "structures without a cell."
                )
        if self.batch_mode == "native":
            totals, energies, forces, virials = self._calculate_native(
                structures, compute_stress
            )
        else:
            totals, energies, forces, virials = self._calculate_sequential(
                structures, compute_stress
            )

        results = []
        for index, energy in enumerate(totals):
            result = {
                "energy": float(energy),
                "free_energy": float(energy),
                "energies": energies[index],
                "forces": forces[index],
            }
            if compute_stress:
                result["stress"] = full_3x3_to_voigt_6_stress(
                    virials[index] / volumes[index]
                )
            results.append(result)
        return results

    def _calculate_sequential(self, structures, compute_stress):
        properties = ["energy", "forces"]
        if compute_stress:
            properties.append("stress")
        totals, energies, forces, virials = [], [], [], []
        for atoms in structures:
            self.calculator.calculate(atoms, properties, all_changes)
            results = self.calculator.results
            totals.append(results["energy"])
            energies.append(np.array(results["energies"], dtype=np.float64))
            forces.append(np.array(results["forces"], dtype=np.float64))
            if compute_stress:
                stress = _voigt_to_3x3(results["stress"])
                virials.append(stress * abs(atoms.cell.volume))
        return totals, energies, forces, virials

    def _calculate_native(self, structures, compute_stress):
        calculator = self.calculator
        evaluator = calculator.evaluator
        atom_counts = np.array([len(atoms) for atoms in structures], dtype=np.int64)
        atom_offsets = np.concatenate(([0], np.cumsum(atom_counts)))
        num_nodes = int(atom_offsets[-1])
        graphs = [self._neighbors(atoms) for atoms in structures]
        edge_offsets = np.concatenate(
            ([0], np.cumsum([len(graph[0]) for graph in graphs]))
        ).astype(np.int64)
        i_list = np.concatenate(
            [graph[0] + offset for graph, offset in zip(graphs, atom_offsets)]
        ).astype(np.int32)
        j_list = np.concatenate(
            [graph[1] + offset for graph, offset in zip(graphs, atom_offsets)]
        ).astype(np.int32)
        r = np.concatenate([graph[2] for graph in graphs])
        xyz = np.concatenate([graph[3] for graph in graphs])
        node_types = self._node_types(
            np.concatenate([atoms.numbers for atoms in structures])
        )
        num_neigh = np.bincount(i_list, minlength=num_nodes)

        properties = ["energy", "forces"]
        if compute_stress:
            properties.append("stress")
        calculator._apply_kernel_launch_tuning(num_nodes, len(i_list), properties)
        mace_inputs = (
            num_nodes,
            node_types.tolist(),
            num_neigh,
            j_list,
            node_types[j_list].tolist(),
            xyz,
            r,
            i_list,
        )
        graph_generation = calculator._compute_mace(mace_inputs, atoms=None)

        # Graph-wide observables accumulate in FP64, as in the ASE calculator.
        node_energies = np.array(evaluator.node_energies, dtype=np.float64, copy=True)[
            :num_nodes
        ]
        pair_forces = None
        reduce_atom_forces = getattr(evaluator, "_reduce_atom_forces", None)
        if callable(reduce_atom_forces):
            forces = np.asarray(
                reduce_atom_forces(num_nodes, i_list, j_list, graph_generation),
                dtype=np.float64,
            ).reshape(num_nodes, 3)
        else:
            pair_forces = _pair_forces(evaluator, len(i_list))
            forces = np.stack(
                [
                    np.bincount(j_list, pair_forces[:, c], minlength=num_nodes)
                    - np.bincount(i_list, pair_forces[:, c], minlength=num_nodes)
                    for c in range(3)
                ],
                axis=1,
            )
        virials = None
        if compute_stress:
            virials = _reduce_virials(
                evaluator, edge_offsets, xyz, graph_generation, pair_forces
            )
        totals = np.bincount(
            np.repeat(np.arange(len(structures)), atom_counts),
            weights=node_energies,
            minlength=len(structures),
        )
        energies = np.split(node_energies, atom_offsets[1:-1])
        forces = np.split(forces, atom_offsets[1:-1])
        return totals, energies, forces, virials

    def _neighbors(self, atoms):
        """Return receiver-major ``(i, j, r, xyz)`` edges of one structure."""
        digest = hashlib.blake2b(digest_size=16)
        for array in (atoms.positions, atoms.cell.array, atoms.pbc):
            digest.update(np.ascontiguousarray(array).tobytes())
        key = digest.digest()
        cached = self._neighbor_cache.get(key)
        if cached is not None:
            self._neighbor_cache.move_to_end(key)
            return cached
        cutoff = self.calculator.cutoff
        if not np.any(atoms.pbc) and atoms.cell.volume == 0.0:
            # A non-periodic cell only bounds the neighbor search.
            atoms = atoms.copy()
            atoms.set_cell(np.ptp(atoms.positions, axis=0) + 2.0 * cutoff + 1.0)
        receivers, sources, shifts = neighbor_list("ijS", atoms, cutoff)
        receivers, sources, shifts = _receiver_major_neighbor_arrays(
            receivers, sources, shifts
        )
        positions = np.asarray(atoms.positions, dtype=np.float64)
        xyz = np.ascontiguousarray(
            positions[sources] - positions[receivers] + shifts @ atoms.cell.array
        )
        r = np.linalg.norm(xyz, axis=1)
        cached = (receivers, sources, r, xyz)
        if self.neighbor_cache_size > 0:
            self._neighbor_cache[key] = cached
            while len(self._neighbor_cache) > self.neighbor_cache_size:
                self._neighbor_cache.popitem(last=False)
        return cached

    def _node_types(self, numbers):
        numbers = np.asarray(numbers, dtype=np.int64)
        known = (numbers >= 0) & (numbers < len(self._type_lookup))
        types = np.full(len(numbers), -1, dtype=np.int64)
        types[known] = self._type_lookup[numbers[known]]
        if np.any(types < 0):
            unsupported = sorted(set(numbers[types < 0].tolist()))
            supported = np.flatnonzero(self._type_lookup >= 0).tolist()
            raise ValueError(
                f"Model does not support atomic numbers {unsupported}. "
                f"Supported atomic numbers are {supported}."
            )
        return types


def _native_unsupported_reason(calculator):
    if calculator._has_native_field_coupling():
        return "electric-field (MACEField) models"
    if getattr(calculator, "_dispersion_calculator", None) is not None:
        return "dispersion corrections"
    if calculator._fixed_workspace_plan_selected():
        return "fixed-workspace tiled plans"
    return None


def _reduce_virials(evaluator, edge_offsets, xyz, graph_generation, pair_forces):
    """Return the virial, ``-sum(F_e outer xyz_e)``, of every edge segment."""
    num_structures = len(edge_offsets) - 1
    # The native reductions divide by a volume; unit volumes return virials.
    unit_volumes = np.ones(num_structures)
    reduce_segmented = getattr(evaluator, "_reduce_segmented_stress", None)
    if callable(reduce_segmented):
        return np.asarray(
            reduce_segmented(unit_volumes, edge_offsets, xyz), dtype=np.float64
        ).reshape(num_structures, 3, 3)
    prepare_batch = getattr(evaluator, "_prepare_factorized_batch", None)
    reduce_batched = getattr(evaluator, "_reduce_batched_stress", None)
    if graph_generation and callable(prepare_batch) and callable(reduce_batched):
        prepare_batch(graph_generation, edge_offsets)
        return np.asarray(
            reduce_batched(unit_volumes, graph_generation), dtype=np.float64
        ).reshape(num_structures, 3, 3)
    if pair_forces is None:
        pair_forces = _pair_forces(evaluator, len(xyz))
    edge_structure = np.repeat(np.arange(num_structures), np.diff(edge_offsets))
    edge_virials = pair_forces[:, :, None] * xyz[:, None, :]
    virials = np.empty((num_structures, 3, 3))
    for row in range(3):
        for column in range(3):
            virials[:, row, column] = -np.bincount(
                edge_structure,
                weights=edge_virials[:, row, column],
                minlength=num_structures,
            )
    return virials


def _pair_forces(evaluator, num_edges):
    pair_forces = np.asarray(evaluator.node_forces, dtype=np.float64).reshape(-1, 3)
    return np.array(pair_forces[:num_edges], copy=True)


def _voigt_to_3x3(stress):
    stress = np.asarray(stress, dtype=np.float64)
    if stress.shape == (3, 3):
        return stress
    xx, yy, zz, yz, xz, xy = stress
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
