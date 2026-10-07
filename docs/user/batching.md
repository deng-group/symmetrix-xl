# Batched evaluation

`Symmetrix.calculate_batch` evaluates a list of independent ASE structures and returns one ASE-style result dictionary per structure, with `energy`, `free_energy`, `energies`, `forces`, and, when requested, `stress` in ASE's Voigt order and sign convention.

```python
from symmetrix import Symmetrix

calc = Symmetrix("mace-omat-0-medium.json", dtype="float32")
results = calc.calculate_batch(structures, properties=("energy", "forces", "stress"))
energies = [result["energy"] for result in results]
```

`batch_mode="native"` (default) places all structures in one disconnected neighbor graph and evaluates it with a single native call. MACE message passing runs only along graph edges, so the structures do not interact. Node energies are summed per structure, and the per-structure stress is reduced natively from the directed pair forces of each structure's edge segment. `batch_mode="sequential"` evaluates the structures one at a time through the ASE calculator path and supports every calculator feature, including dispersion corrections and MACEField models, which the native mode rejects.

`symmetrix.BatchEvaluator(calc, batch_mode="native", neighbor_cache_size=64)` exposes the same evaluation with an explicit neighbor-list cache. Neighbor lists are keyed by positions, cell, and periodicity, so structures that differ only in species, such as the orderings of a disordered lattice before relaxation, share one neighbor list. Stress requires three-dimensional cells; omit it for molecules without a cell.

## Performance

A native batch pays the fixed cost of an evaluation (kernel submissions, synchronization, and host bookkeeping) once per batch instead of once per structure. It therefore helps most when a single small structure leaves a GPU underused. On one CPU core the cost already scales with the edge count, and batching saves only the per-call overhead.

`benchmarks/batch_evaluation_benchmark.py` evaluates B-site orderings of a 40-atom SrTi0.5Zr0.5O3 supercell (2x2x2 cubic perovskite, a = 4.0 A) with MACE-OMAT-0 medium restricted to O, Ti, Sr, and Zr. The run below used FP32 arithmetic, a 6.0 A model cutoff without a neighbor skin, and 2,480 directed edges per structure, on one core of an Apple M1 Max. Times are steady-state medians of one batch evaluation, including the neighbor graph, native evaluation, and reductions of energies, forces, and stresses.

| Batch | Atoms | Native (us/atom) | Sequential (us/atom) |
| ---: | ---: | ---: | ---: |
| 1 | 40 | 496 | 526 |
| 16 | 640 | 499 | 512 |
| 64 | 2,560 | 503 | 522 |

The same driver accepts `--reference-mace-model` to time the batched MACE model of torch-sim on the same structures as an external reference. On this core it took 4,100 to 6,400 us/atom.
