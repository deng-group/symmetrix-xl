import json
import os
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress

from symmetrix import BatchEvaluator, Symmetrix
from symmetrix import symmetrix as native_symmetrix

SPECIES = [8, 22, 38, 40]


@pytest.fixture(scope="module")
def omat_small_model():
    foundations = pytest.importorskip(
        "mace.calculators.foundations_models",
        reason="mace-torch is required to download the foundation model",
    )
    from model_downloads import test_model_cache_dir

    cache_dir = test_model_cache_dir() / "mace-foundation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    try:
        return str(foundations.download_mace_mp_checkpoint("small-omat-0"))
    except Exception as exc:
        pytest.skip(f"MACE foundation model is not available: {exc}")
    finally:
        if previous is None:
            del os.environ["XDG_CACHE_HOME"]
        else:
            os.environ["XDG_CACHE_HOME"] = previous


@pytest.fixture(scope="module")
def fp64_calculator(omat_small_model):
    return Symmetrix(omat_small_model, species=SPECIES, dtype="float64")


def _perovskite(repeat=(2, 2, 2), a=3.95):
    return Atoms(
        "SrTiO3",
        scaled_positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ],
        cell=[a, a, a],
        pbc=True,
    ).repeat(repeat)


def _orderings(count, seed=0):
    """B-site orderings of SrTi0.5Zr0.5O3 that share one geometry."""
    lattice = _perovskite()
    b_sites = np.flatnonzero(lattice.numbers == 22)
    rng = np.random.default_rng(seed)
    structures = []
    for _ in range(count):
        atoms = lattice.copy()
        atoms.numbers[rng.choice(b_sites, len(b_sites) // 2, replace=False)] = 40
        structures.append(atoms)
    return structures


def _mixed_structures():
    first, second = _orderings(2)
    rattled = second.copy()
    rattled.rattle(0.05, seed=3)
    # Atoms outside the cell must give the same graph as their periodic images.
    rattled.positions[:3] += rattled.cell[0] - 2.0 * rattled.cell[2]
    sheared = _perovskite((2, 1, 1))
    sheared.set_cell(sheared.cell.array + [[0, 0, 0], [0.4, 0, 0], [0, 0.3, 0]])
    sheared.rattle(0.03, seed=4)
    return [first, rattled, sheared, first.copy()]


def _ase_reference(atoms, calculator):
    atoms = atoms.copy()
    atoms.calc = calculator
    return atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()


def _assert_matches_ase(
    results, structures, calculator, energy_atol, force_atol, stress_atol
):
    assert len(results) == len(structures)
    for result, atoms in zip(results, structures):
        energy, forces, stress = _ase_reference(atoms, calculator)
        np.testing.assert_allclose(result["energy"], energy, rtol=0.0, atol=energy_atol)
        np.testing.assert_allclose(result["forces"], forces, rtol=0.0, atol=force_atol)
        if "stress" in result:
            np.testing.assert_allclose(
                result["stress"], stress, rtol=0.0, atol=stress_atol
            )


@pytest.mark.parametrize("batch_mode", ["native", "sequential"])
def test_batch_matches_ase_calculator(fp64_calculator, batch_mode):
    structures = _mixed_structures()

    results = fp64_calculator.calculate_batch(
        structures, properties=("energy", "forces", "stress"), batch_mode=batch_mode
    )

    for result, atoms in zip(results, structures):
        assert result["forces"].shape == (len(atoms), 3)
        assert result["energies"].shape == (len(atoms),)
        assert result["stress"].shape == (6,)
        assert result["free_energy"] == result["energy"]
        np.testing.assert_allclose(
            np.sum(result["energies"]), result["energy"], rtol=0.0, atol=1e-10
        )
    _assert_matches_ase(results, structures, fp64_calculator, 1e-9, 1e-9, 1e-10)
    assert results[3]["energy"] == results[0]["energy"]


def test_native_segmented_stress_matches_pair_force_reduction(fp64_calculator):
    evaluator = fp64_calculator.evaluator
    if not callable(getattr(evaluator, "_reduce_segmented_stress", None)):
        pytest.skip("the evaluator has no native segmented stress reduction")
    structures = _mixed_structures()
    batch = BatchEvaluator(fp64_calculator)
    batch.calculate(structures, properties=("energy", "forces", "stress"))
    graphs = [batch._neighbors(atoms) for atoms in structures]
    edge_offsets = np.concatenate(([0], np.cumsum([len(g[0]) for g in graphs])))
    xyz = np.concatenate([graph[3] for graph in graphs])
    pair_forces = np.asarray(evaluator.node_forces, dtype=float).reshape(-1, 3)
    volumes = np.array([atoms.get_volume() for atoms in structures])

    native = evaluator._reduce_segmented_stress(volumes, edge_offsets, xyz)

    assert native.shape == (len(structures), 3, 3)
    for index in range(len(structures)):
        edges = slice(edge_offsets[index], edge_offsets[index + 1])
        expected = -(pair_forces[edges].T @ xyz[edges]) / volumes[index]
        np.testing.assert_allclose(native[index], expected, rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match="inconsistent"):
        evaluator._reduce_segmented_stress(volumes[:-1], edge_offsets, xyz)
    with pytest.raises(ValueError, match="invalid"):
        evaluator._reduce_segmented_stress(-volumes, edge_offsets, xyz)


def test_orderings_share_one_neighbor_list(fp64_calculator):
    structures = _orderings(6)
    batch = BatchEvaluator(fp64_calculator)

    results = batch.calculate(structures, properties=("energy", "forces", "stress"))

    assert len(batch._neighbor_cache) == 1
    _assert_matches_ase(results[:2], structures[:2], fp64_calculator, 1e-9, 1e-9, 1e-10)
    # Species changes on a cached geometry must not reuse stale node types.
    swapped = [atoms.copy() for atoms in structures]
    for atoms in swapped:
        atoms.numbers[atoms.numbers == 38] = 40
    swapped_results = batch.calculate(swapped, properties=("energy", "forces"))
    assert len(batch._neighbor_cache) == 1
    _assert_matches_ase(
        swapped_results[:1], swapped[:1], fp64_calculator, 1e-9, 1e-9, 0
    )


def test_neighbor_cache_size_bounds_entries(fp64_calculator):
    structures = _mixed_structures()
    properties = ("energy", "forces", "stress")
    uncached = BatchEvaluator(fp64_calculator, neighbor_cache_size=0)
    bounded = BatchEvaluator(fp64_calculator, neighbor_cache_size=2)

    reference = uncached.calculate(structures, properties)
    results = bounded.calculate(structures, properties)

    assert len(uncached._neighbor_cache) == 0
    assert len(bounded._neighbor_cache) == 2
    for result, expected in zip(results, reference):
        for key in properties:
            np.testing.assert_array_equal(result[key], expected[key])


def test_batch_rejects_invalid_requests(fp64_calculator):
    with pytest.raises(ValueError, match="batch_mode"):
        BatchEvaluator(fp64_calculator, batch_mode="parallel")
    batch = BatchEvaluator(fp64_calculator)
    assert batch.calculate([]) == []
    with pytest.raises(ValueError, match="Unsupported batch property"):
        batch.calculate(_orderings(1), properties=("energy", "becs"))

    unsupported = _orderings(1)[0]
    unsupported.numbers[0] = 1
    with pytest.raises(ValueError, match=r"atomic numbers \[1\]"):
        batch.calculate([unsupported])

    molecule = Atoms("TiO2", positions=[[0, 0, 0], [1.7, 0, 0], [-1.7, 0, 0]])
    with pytest.raises(ValueError, match="three-dimensional cells"):
        batch.calculate([molecule], properties=("energy", "stress"))
    result = batch.calculate([molecule], properties=("energy", "forces"))[0]
    assert "stress" not in result
    # The ASE calculator inverts the cell, so its reference uses a vacuum box.
    boxed = molecule.copy()
    boxed.set_cell([12.0, 12.0, 12.0])
    _assert_matches_ase([result], [boxed], fp64_calculator, 1e-9, 1e-9, 0)


def test_native_batches_reject_dispersion(fp64_calculator):
    fp64_calculator._dispersion_calculator = object()
    try:
        with pytest.raises(ValueError, match="dispersion"):
            BatchEvaluator(fp64_calculator)
        BatchEvaluator(fp64_calculator, batch_mode="sequential")
    finally:
        fp64_calculator._dispersion_calculator = None


def test_native_mh1_batch_matches_ase_calculator(tmp_path):
    from test_mh1 import _make_generalized_mh1

    if not hasattr(native_symmetrix, "MACENonlinearKokkos"):
        pytest.skip("Symmetrix was built without FP64 Kokkos MH-1 support")
    if native_symmetrix._kokkos_default_execution_space() in ("Cuda", "HIP"):
        pytest.skip("generated accelerator MH-1 direct execution requires FP32")
    _, data = _make_generalized_mh1((4, 2, 4, 2, []))
    path = tmp_path / "mh1.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    calculator = Symmetrix(path, use_kokkos=True, dtype="float64")
    structures = [
        Atoms(
            numbers=[14, 1, 14],
            positions=[[0.0, 0.0, 0.0], [1.7, 0.2, 0.1], [0.3, 1.8, 0.4]],
            cell=[7.0, 7.0, 7.0],
            pbc=True,
        ),
        Atoms(
            numbers=[14, 14, 1, 1],
            positions=[
                [0.1, 0.0, 0.0],
                [1.9, 0.3, 0.2],
                [0.4, 1.6, 0.1],
                [2.2, 1.4, 0.5],
            ],
            cell=[6.5, 7.0, 7.5],
            pbc=True,
        ),
    ]

    results = calculator.calculate_batch(structures, ("energy", "forces", "stress"))

    _assert_matches_ase(results, structures, calculator, 1e-9, 1e-9, 1e-10)


def test_native_batch_agrees_with_torch_sim_mace_model(
    fp64_calculator, omat_small_model
):
    torch = pytest.importorskip("torch")
    ts = pytest.importorskip("torch_sim", reason="torch-sim is not installed")
    from torch_sim.models.mace import MaceModel

    structures = _mixed_structures()[:3]
    reference = MaceModel(
        model=torch.load(omat_small_model, weights_only=False, map_location="cpu"),
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_stress=True,
    )(ts.io.atoms_to_state(structures, torch.device("cpu"), torch.float64))

    results = fp64_calculator.calculate_batch(
        structures, properties=("energy", "forces", "stress")
    )

    # Symmetrix evaluates radial functions from splines, so it agrees with
    # MACE-Torch to spline accuracy rather than to rounding.
    offset = 0
    for index, (result, atoms) in enumerate(zip(results, structures)):
        np.testing.assert_allclose(
            result["energy"], reference["energy"][index].item(), rtol=0.0, atol=1e-4
        )
        np.testing.assert_allclose(
            result["forces"],
            reference["forces"][offset : offset + len(atoms)].numpy(),
            rtol=0.0,
            atol=5e-4,
        )
        np.testing.assert_allclose(
            result["stress"],
            full_3x3_to_voigt_6_stress(reference["stress"][index].numpy()),
            rtol=0.0,
            atol=1e-5,
        )
        offset += len(atoms)


def _metal_available():
    if sys.platform != "darwin":
        return False, "Metal requires macOS"
    if not getattr(native_symmetrix, "_metal_supported", lambda: False)():
        return False, "the native extension was built without SYMMETRIX_METAL"
    return native_symmetrix._metal_device_ready()


def test_native_metal_batch_matches_host_fp32(omat_small_model):
    available, reason = _metal_available()
    if not available:
        pytest.skip(reason)
    host = Symmetrix(omat_small_model, species=SPECIES, dtype="float32")
    metal = Symmetrix(omat_small_model, species=SPECIES, dtype="float32", metal=True)
    structures = _mixed_structures()

    results = metal.calculate_batch(structures, ("energy", "forces", "stress"))

    assert metal.metal_status == "ready"
    assert metal.metal_statistics()["forward_launches"] >= 1
    # FP32 summation order differs between the host owners and the GPU.
    _assert_matches_ase(results, structures, host, 1e-5 * 80, 1e-4, 1e-5)
