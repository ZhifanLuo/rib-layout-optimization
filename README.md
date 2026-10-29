# Rib layout optimization - v1.7.0

Python implementation for *An Active-Set Framework for Explicit Stiffening-Rib
Layout Optimization*, with five examples and the comparisons used in the revised
manuscript. This release includes the 1.0 mm-base Example V, density-based
topology optimization and evaluation of the rib layouts in the same shell-solid
model. Earlier tags remain unchanged.

## Install and test

Use Python 3.12. From the repository root:

```console
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -B -m unittest discover -s tests -v
```

On Linux/macOS, activate the environment and use `python` or `.venv/bin/python`.
Windows users can also run `setup_env.bat`. Pardiso/MKL is used when available;
SciPy SuperLU is the fallback. Large solid models need substantial memory and
compute time. Solver and library differences can change execution time and the
trajectory of a nonconvex optimization.

## Run Examples I-V

All five example scripts use the same CLI and output schema:

```console
python -B example1.py --output ../rerun-I
python -B example2.py --output ../rerun-II
python -B example3.py --output ../rerun-III
python -B example4.py --output ../rerun-IV
python -B example5.py --output ../rerun-V
```

Each output root receives an `example_N/` directory. Use fresh output roots.
Omit `--quick` and iteration-budget overrides for paper calculations.
`example_all.bat` retains its original sequential I-IV scope; run V separately.

Example V uses a 100 x 50 mm plate, 1.0 mm base, 80 x 40 base mesh, 5 mm rib-top
height measured from the base midsurface, and -0.0001 N/mm2 uniform pressure.
Only the right base edge is clamped. Its complete normalized input is
`configs/paper/example5.json`. The saved run ends with six ribs and two
centre-line crossings, after 158 response evaluations. To plan or reproduce it:

```console
python -B tools/reproduce_paper.py --case 5 --output ../paper-V
python -B tools/reproduce_paper.py --case 5 --output ../paper-V --execute
```

The helper requires a new output directory outside the source tree, uses the
standard iteration budgets, and records source, input and result hashes.

## Method and comparison models

The rib optimizer uses explicit MITC4/Reissner-Mindlin shell ribs and a fixed
shell base. Rib roots follow base interpolation; crossing ribs transfer forces
through the base. Direct rib junctions are checked separately. Native rib volume
is additive strip volume. See `tools/REVISION_EXPERIMENTS.md` for controlled
addition, deletion and rationalization studies.

The Python density method in `rib_layout_algorithms/density_solid.py` uses a
fixed shell base and eight-node 3D solid elements above it. One density variable
per vertical column gives an equal-height extrusion. SIMP, density filtering
and OC updates minimize compliance under an added-material volume limit. The
underlying references are supplied in `references/` and
[the model documentation](docs/EXTRUDED_DENSITY_COMPARISON.md).

The selected comparison uses Examples III and V. The saved rib layouts are
converted to actual-width solid geometry and reanalyzed with the same fixed
base mesh, loads, supports and shell-solid interface as the density layouts.
Overlaps are counted once, and density volume limits match the added solid rib
volumes. These responses differ from the native rib-shell responses because
the element types, base mesh and rib connections differ. Continuous density
fields are compared; no binary-density comparison is used.

For a read-only generic density-run plan covering any of I-V:

```console
python -B tools/run_extruded_density_comparison.py --cases 3 5 --mesh 160 80 --vertical-elements 8 --output ../density-plan
```

Append `--execute` to run a new optimization. The generic driver uses the
native example volume budget unless supplied with the separately prepared
paper inputs. Use the release-data commands below to replay the published
matched-volume comparisons, including their exact numerical adapters.

`density_shell.py` and `docs/DENSITY_COMPARISON.md` retain the earlier
parallel-layer model and shared filter/OC implementation. That layer model is
not the comparison reported in the current paper.

## Download the saved results

Release: https://github.com/ZhifanLuo/rib-layout-optimization/releases/tag/v1.7.0

Download `rib-layout-optimization-v1.7.0-revision-data.zip` and check it against
`SHA256SUMS.txt`. It supplies the versioned `Code/` snapshot, I-IV results and
controlled studies, the accepted revised V, III/V density fields, common-model
rib responses, junction checks, regular six/eight-rib references and figure
sources. Start with its root README and evidence index. The nested
`Supplementary_Density_Comparison.zip` is an unchanged, checksummed snapshot
with its exact executed sources and portable reproduction commands. Its
pre-release notes describe its creation history; the v1.7.0 root index defines
which results support the current paper.

Saved results retain their actual input configurations, source fingerprints
and execution dates. Packaging does not rerun optimization or relabel older
results. The 0.5 mm-base V belongs to v1.6.0; the current V has a 1.0 mm base.
Historical `verification/`, `verification_v1_5_0/`, and
`configs/reported_results.json` are records or comparison targets, not solver
inputs. `configs/examples.yaml` is archival.

See [the reproduction guide](docs/REPRODUCTION.md) for the extraction layout,
dependencies, current-result summary and exact replay entry points.

## License and citation

Project Python source is BSD-3-Clause; use `CITATION.cff` to cite this version
and cite the associated paper. Third-party components retain their licenses
and attribution. References establish the numerical ingredients; the column
mapping, shell-solid coupling and rib conversion are implementation choices.
No new DOI is asserted for this release.
