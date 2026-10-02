# Apple Metal (macOS)

The `metal` branch adds an Apple GPU backend for FP32 MACE evaluation on Apple silicon Macs. The radial, product-basis, and interaction stages run as native Metal kernels on the GPU, and the remaining stages run on one CPU core. It is a source build: there is no wheel yet, and the maintained build helper `tools/symmetrix_build.py` does not have a Metal option.

## Requirements

- An Apple silicon Mac (M1 or newer) on macOS 13 or newer. Intel Macs are not supported.
- Xcode Command Line Tools, which provide the Clang compiler and the Metal runtime compiler. The full Xcode application is not required.
- Homebrew `gcc`, which provides `gfortran`; the CPU host build enables Fortran.
- `git` and [`uv`](https://docs.astral.sh/uv/).

```bash
xcode-select --install          # skip if already installed
brew install gcc uv
```

## Build and install

The commands below clone the branch with its submodules, create a fresh virtual environment, and build the extension with the Metal layer enabled. The build takes a few minutes and installs `mace-torch`, which is needed only to convert models.

```bash
git clone --recursive --branch metal https://github.com/deng-group/symmetrix-xl.git
cd symmetrix-xl
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "./symmetrix[mace]" \
  -C cmake.define.SYMMETRIX_METAL=ON \
  -C cmake.define.Kokkos_ENABLE_SERIAL=ON \
  -C cmake.define.Kokkos_ENABLE_OPENMP=OFF \
  -C cmake.define.CMAKE_Fortran_COMPILER="$(brew --prefix)/bin/gfortran" \
  -C cmake.define.Python_EXECUTABLE="$(which python)" \
  -C cmake.define.PYTHON_EXECUTABLE="$(which python)" \
  -C build-dir=build-metal
```

The host side uses the Kokkos Serial backend. An OpenMP host build also works, but the Homebrew OpenMP runtime conflicts with the one bundled in PyTorch when both are imported in the same process, which aborts with `OMP: Error #15`.

To rebuild after pulling new commits, activate the environment and rerun the same `uv pip install` command with `--reinstall-package symmetrix-xl`.

## Convert a model

Symmetrix evaluates a JSON export of a MACE model. For example, for MACE-MP-0b medium:

```bash
curl -L -o mace_agnesi_medium.model \
  https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model
symmetrix_extract_mace --model mace_agnesi_medium.model --output mace_agnesi_medium.json
```

## Run

Pass `metal=True` together with `dtype="float32"`; Metal has no FP64 arithmetic, so the backend is FP32 only.

```python
from ase.build import bulk
from symmetrix import Symmetrix

atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((6, 6, 6))
atoms.rattle(0.05, seed=1)
calc = Symmetrix("mace_agnesi_medium.json", dtype="float32", metal=True)
print(calc.metal_status, calc.metal_device)   # ready  Apple M1 Max
atoms.calc = calc
print(atoms.get_potential_energy())
print(atoms.get_forces()[:3])
```

`Symmetrix(..., metal=True)` raises an error if the Metal layer is not available, so a run cannot fall back to the CPU silently. `calc.metal_statistics()` reports per-stage GPU launch counts and times. Any ASE workflow works with this calculator, including molecular dynamics; the `metal=True` and `dtype="float32"` arguments are the only change from the CPU calculator.

## Check the installation

```bash
python -m pytest symmetrix/test/test_metal_execution.py
```

The GPU parity tests download the `small-omat-0` foundation model on first use and compare Metal energies, forces, and stresses against the FP32 CPU path.

## Expected performance and accuracy

On an M1 Max with MACE-MP-0b medium (2000-atom Na3SbS4, 6.0 A cutoff with a 0.5 A neighbor skin, 110,000 directed edges), 1000 NVT MD steps take 38 s, or 19 us/atom/step. The trajectory stays within 3 meV in total energy of an FP64 reference run over all 1000 steps.

Benchmarks on a laptop are sensitive to other GPU and CPU load. Animated wallpapers, video playback, Spotlight indexing, and security scanners noticeably slow the GPU stages; close them for timing runs.

## Limitations

- FP32 only.
- The Metal R0 stage supports spherical harmonics up to `l_max` 3, which covers the MACE-MP-0 and OMAT-0 medium models. The fastest R1 edge kernel needs a channel count that is a multiple of 32; other channel counts use a slower per-edge kernel.
- One process uses one GPU; there is no LAMMPS or MPI support for the Metal backend yet.
- The first evaluation in a process compiles the GPU kernels, which takes a fraction of a second.
