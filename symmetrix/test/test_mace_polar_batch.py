"""Batched MACE-POLAR-1 evaluation of independent structures.

``Symmetrix.calculate_polar_batch`` evaluates several structures in one native
transaction: one joined backbone graph whose segment geometry keeps every
structure in its own cell, and one long-range program with a graph per
structure. Each structure must receive the result of an independent
evaluation, which these tests check against ``calculate`` on each structure.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from symmetrix.calculator import Symmetrix
from symmetrix.mace_polar_frontend import MacePolarFrontendAdapter
from test_mace_polar_frontend import (  # noqa: F401 (pytest fixtures)
    polar_jit_cache,
    polar_model_path,
    polar_models,
)

PROPERTIES = ["energy", "forces", "stress", "dipole"]
# (absolute, relative) agreement with an FP64 single evaluation. A single FP32
# evaluation of this model differs from FP64 by up to 1e-3 eV/A in force on an
# M1 Max and an RTX 4090; an FP32 batch, which sums in a different order, stays
# within the same bound.
TOLERANCE = {"float64": (1e-10, 0.0), "float32": (2e-3, 1e-3)}
# A batch passes the edges inside the cutoff. A single evaluation also passes
# the neighbor-skin shell clamped to the cutoff, where the radial functions of
# trained models vanish (MACE-POLAR-1-S gives bit-identical results for any
# skin) but those of the untrained test model do not. The references therefore
# use a negligible skin.
REFERENCE_SKIN = 1e-6


class _Source:
    """Trajectory identity, as atombatch attaches it to each request."""

    __slots__ = ("__weakref__",)


def _cluster(seed, count, **info):
    rng = np.random.default_rng(seed)
    positions = [rng.uniform(0.0, 1.0, 3)]
    while len(positions) < count:
        candidate = positions[-1] + rng.normal(0.0, 1.0, 3) * 1.4
        if min(np.linalg.norm(candidate - p) for p in positions) > 1.3:
            positions.append(candidate)
    symbols = ["Si" if index % 2 == 0 else "H" for index in range(count)]
    atoms = Atoms(symbols, positions=positions, pbc=False)
    atoms.center(vacuum=5.0)
    atoms.info.update(info)
    return atoms


def _crystal(seed, repeat, **info):
    rng = np.random.default_rng(seed)
    atoms = Atoms(
        "SiH",
        scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        cell=np.diag([3.4, 3.6, 3.8]),
        pbc=True,
    ).repeat(repeat)
    atoms.rattle(0.05, seed=seed)
    atoms.set_cell(atoms.cell.array + rng.normal(0.0, 0.05, (3, 3)), scale_atoms=True)
    atoms.info.update(info)
    return atoms


def _structures():
    return [
        _cluster(1, 3),
        _crystal(2, (1, 1, 2)),
        _cluster(3, 5, total_charge=1.0, total_spin=1.0),
        _crystal(4, (2, 1, 1), electric_field=[0.0, 0.0, 0.02]),
        _cluster(5, 4, electric_field=[0.01, 0.0, 0.0]),
        _crystal(6, (1, 2, 2)),
    ]


def _single(calculator, atoms, properties):
    calculator.reset()
    calculator.calculate(atoms.copy(), properties)
    return {name: calculator.results[name] for name in properties}


@pytest.mark.parametrize("dtype", ("float64", "float32"))
@pytest.mark.parametrize("route", ("reference", "direct"))
def test_batch_matches_independent_evaluations(polar_model_path, dtype, route):
    calculator = Symmetrix(str(polar_model_path), dtype=dtype, streamed_edges=route)
    reference = Symmetrix(
        str(polar_model_path),
        dtype="float64",
        streamed_edges=route,
        neighbor_skin=REFERENCE_SKIN,
    )
    structures = _structures()
    for atoms in structures:
        atoms.info["atombatch_source"] = _Source()
    batches_before = calculator.evaluator.polar_whole_model_batch_count

    results = calculator.calculate_polar_batch(structures, PROPERTIES)
    # A second call after every atom moved reuses each trajectory's topology.
    for atoms in structures:
        atoms.positions += 0.01
    results = calculator.calculate_polar_batch(structures, PROPERTIES)

    # Isolated and periodic structures form two native calls per batch.
    assert calculator.evaluator.polar_whole_model_batch_count - batches_before == 4
    absolute, relative = TOLERANCE[dtype]
    for atoms, result in zip(structures, results):
        expected = _single(reference, atoms, PROPERTIES)
        assert result["energy"] == pytest.approx(
            expected["energy"], abs=absolute * len(atoms), rel=relative
        )
        assert result["free_energy"] == result["energy"]
        for name in ("forces", "stress", "dipole"):
            np.testing.assert_allclose(
                result[name], expected[name], rtol=relative, atol=absolute
            )


def test_single_evaluation_after_batch_matches_fresh_calculator(polar_model_path):
    calculator = Symmetrix(
        str(polar_model_path), dtype="float64", neighbor_skin=REFERENCE_SKIN
    )
    structures = _structures()
    calculator.calculate_polar_batch(structures, ["energy", "forces"])
    fresh = Symmetrix(
        str(polar_model_path), dtype="float64", neighbor_skin=REFERENCE_SKIN
    )
    for atoms in (structures[1], structures[2]):
        after = _single(calculator, atoms, PROPERTIES)
        expected = _single(fresh, atoms, PROPERTIES)
        for name in PROPERTIES:
            np.testing.assert_allclose(
                after[name], expected[name], rtol=0.0, atol=1e-12
            )


def test_batch_returns_only_requested_properties(polar_model_path):
    calculator = Symmetrix(str(polar_model_path), dtype="float64")
    results = calculator.calculate_polar_batch(_structures()[:2], ["energy"])
    assert [sorted(result) for result in results] == [["energy", "free_energy"]] * 2


def test_native_batch_requires_one_periodicity(polar_model_path):
    calculator = Symmetrix(str(polar_model_path), dtype="float64")
    mixed = [_cluster(1, 3), _crystal(2, (1, 1, 2))]
    with pytest.raises(ValueError, match="same PBC pattern"):
        MacePolarFrontendAdapter.evaluate_batch(calculator, mixed, ["energy"])
    # The public method groups by periodicity instead.
    assert len(calculator.calculate_polar_batch(mixed, ["energy"])) == 2
