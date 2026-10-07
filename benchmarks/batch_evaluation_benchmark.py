"""Benchmark batched evaluation of small structures.

The workload is a set of B-site orderings of a cubic perovskite supercell
(SrTi1-xZrxO3 by default), the typical input of a configurational screen. Each
batch size is evaluated by ``BatchEvaluator`` in native and sequential batch
modes and, optionally, by the batched MACE model of torch-sim as an external
reference. The record reports steady-state us/atom per evaluation
together with the atom count, model cutoff, and directed-edge count of each
structure.
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import statistics
import sys
import time

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _orderings(count, repeat, substituted, seed, rattle):
    from ase import Atoms

    a = 4.0
    primitive = Atoms(
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
    )
    supercell = primitive.repeat((repeat,) * 3)
    b_sites = [i for i, z in enumerate(supercell.numbers) if z == 22]
    rng = np.random.default_rng(seed)
    structures = []
    for index in range(count):
        atoms = supercell.copy()
        atoms.numbers[rng.choice(b_sites, substituted, replace=False)] = 40
        if rattle > 0.0:
            atoms.rattle(rattle, seed=seed + index)
        structures.append(atoms)
    return structures


def _time(operation, synchronize, warmups, repeats):
    for _ in range(warmups):
        operation()
    synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        synchronize()
        samples.append(time.perf_counter() - start)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Symmetrix JSON or MACE model.")
    parser.add_argument(
        "--reference-mace-model",
        help="MACE checkpoint evaluated by torch-sim's MACE model as a reference.",
    )
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--metal", action="store_true")
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--substituted", type=int, default=4)
    parser.add_argument("--rattle", type=float, default=0.0)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--batch-modes", nargs="+", default=["native", "sequential"])
    parser.add_argument(
        "--neighbor-cache-size",
        type=int,
        default=64,
        help="Neighbor lists retained by native batches; 0 rebuilds them every call.",
    )
    parser.add_argument(
        "--properties", nargs="+", default=["energy", "forces", "stress"]
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    import symmetrix
    from symmetrix import BatchEvaluator, Symmetrix

    native = symmetrix.load_backend()
    extension = pathlib.Path(native.__file__).resolve()
    model_kwargs = {"dtype": args.dtype}
    if args.metal:
        model_kwargs["metal"] = True
    # The ASE path keeps neighbor-skin edges in direct execution; without a
    # skin both modes evaluate the same directed graph.
    calculator = Symmetrix(args.model, neighbor_skin=0.0, **model_kwargs)
    properties = tuple(args.properties)
    methods = {}
    for batch_mode in args.batch_modes:
        evaluator = BatchEvaluator(
            calculator,
            batch_mode=batch_mode,
            neighbor_cache_size=args.neighbor_cache_size,
        )
        methods[f"symmetrix-{batch_mode}"] = (
            lambda structures, evaluator=evaluator: evaluator.calculate(
                structures, properties
            ),
            lambda: None,
        )
    if args.reference_mace_model:
        import torch
        import torch_sim
        from torch_sim.models.mace import MaceModel

        device = torch.device(args.reference_device)
        dtype = torch.float64 if args.dtype == "float64" else torch.float32
        reference = MaceModel(
            model=torch.load(
                args.reference_mace_model, weights_only=False, map_location=device
            ),
            device=device,
            dtype=dtype,
            compute_stress="stress" in properties,
        )

        def reference_call(structures):
            return reference(torch_sim.io.atoms_to_state(structures, device, dtype))

        def reference_synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        methods["torch-sim-mace"] = (reference_call, reference_synchronize)

    structures = _orderings(
        max(args.batch_sizes), args.repeat, args.substituted, args.seed, args.rattle
    )
    from ase.neighborlist import neighbor_list

    edges = [len(neighbor_list("i", atoms, calculator.cutoff)) for atoms in structures]
    record = {
        "workload": {
            "structure": f"SrTi1-xZrxO3 {args.repeat}x{args.repeat}x{args.repeat}",
            "atoms_per_structure": len(structures[0]),
            "substituted_b_sites": args.substituted,
            "rattle_A": args.rattle,
            "neighbor_cache_size": args.neighbor_cache_size,
            "properties": list(properties),
            "model_cutoff_A": float(calculator.cutoff),
            "neighbor_skin_A": 0.0,
            "effective_cutoff_A": float(calculator.cutoff),
            "directed_edges_per_structure": sorted(set(edges)),
        },
        "model": {
            "path": str(pathlib.Path(args.model).resolve()),
            "sha256": _sha256(args.model),
        },
        "runtime": {
            "python": sys.executable,
            "symmetrix": str(pathlib.Path(symmetrix.__file__).resolve()),
            "extension": str(extension),
            "extension_sha256": _sha256(extension),
            "kokkos_execution_space": native._kokkos_default_execution_space(),
            "dtype": args.dtype,
            "metal": calculator.metal_status,
            "metal_stages": list(calculator.metal_stages),
            "streamed_edges": calculator.evaluator.streamed_edges_mode,
            "reference_device": args.reference_device
            if args.reference_mace_model
            else None,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "threads_env": {
                name: os.environ.get(name)
                for name in (
                    "KOKKOS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "timing": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "statistic": "median",
        },
        "results": [],
    }
    print(json.dumps({key: record[key] for key in ("workload", "runtime")}, indent=2))
    for batch_size in args.batch_sizes:
        batch = structures[:batch_size]
        atoms = sum(len(item) for item in batch)
        for name, (call, synchronize) in methods.items():
            samples = _time(
                lambda call=call: call(batch), synchronize, args.warmups, args.repeats
            )
            median = statistics.median(samples)
            row = {
                "method": name,
                "batch_size": batch_size,
                "atoms": atoms,
                "directed_edges": int(sum(edges[:batch_size])),
                "median_s": median,
                "us_per_atom": 1e6 * median / atoms,
                "samples_s": samples,
            }
            record["results"].append(row)
            print(
                f"{name:22s} batch {batch_size:4d}  atoms {atoms:6d}  "
                f"{row['us_per_atom']:9.2f} us/atom  {1e3 * median:9.2f} ms/call"
            )
    if args.output:
        args.output.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
