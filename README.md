# Rib layout optimization — v1.6.0

Reference Python implementation for *An Active-Set Framework for Explicit
Stiffening-Rib Layout Optimization*. This release includes the revised five-case
workflow, uniform pressure loading, controlled simplification comparisons and the
two-axis first-group four-rib addition rule.

The base plate and explicit ribs use MITC4/Reissner–Mindlin shells. Rib roots are
tied to the base interpolation; separate rib-top traces remain independent at
crossings. Rib volume is additive strip volume, not a Boolean union. These are
stiffness-focused idealizations, not validated models of cast/welded junctions,
manufacturing, strength or buckling.

## Install and test

Use Python 3.12 and a virtual environment. From the repository root:

```console
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -B -m unittest discover -s tests -v
```

On Linux/macOS activate the environment and use `python` or `.venv/bin/python`.
Windows users can run `setup_env.bat`. MKL/Pardiso availability depends on the
platform; SciPy SuperLU is the fallback. Library/solver differences can affect
execution times and the trajectory of this nonconvex optimization.

## Run the five examples

`example1.py` through `example5.py` share one command-line interface and output
schema. Use the environment's Python from the repository root:

```console
python -B example1.py --output ../rerun-I
python -B example2.py --output ../rerun-II
python -B example3.py --output ../rerun-III
python -B example4.py --output ../rerun-IV
python -B example5.py --output ../rerun-V
```

Each output root receives `example_N/` with results.json, CSV summaries, stage
histories and figures. Choose new roots to retain saved evidence. Omit `--quick`
and `--geometry-sweeps` for paper calculations. On Windows, `example_all.bat`
preserves the original sequential I–IV runner; V is run separately. Older
combined tools retain their original I–IV scope.

Example V now defaults to the manuscript's **100 × 50 mm base, 0.5 mm thickness,
80 × 40 mesh**, 5 mm rib height and −0.0001 N/mm² uniform pressure (−0.5 N total).
Only the base's right edge is clamped; rib-top nodes stay free. The 25 mm initial
lattice supplies 16 ribs, and the rib-volume bound is 1500 mm³. The earlier
1 mm / 40 × 20 model is historical, not the v1.6.0 default.

The complete normalized archived input is `configs/paper/example5.json`, equal
to `load_case(5)`. For a guarded plan and full reproduction:

```console
python -B tools/reproduce_paper.py --output ../paper-V
python -B tools/reproduce_paper.py --output ../paper-V --execute
```

The first command writes nothing. The second runs the exact configuration with
no quick-mode/iteration-cap overrides, reserves a fresh directory outside the
source tree, and records configuration/source/result hashes. Interrupted/failed
output is retained, not overwritten. Separate stage-response replays are excluded
from optimization FEA counts.

## Algorithm and controls

The sparse route alternates sizing, validated filtering and candidate insertion,
then jointly optimizes endpoints and thicknesses. Candidate scores measure locally
equilibrated stiffness contribution per volume in the frozen base field; they
are not actual global compliance changes. Global resizing must confirm at least
1% compliance improvement before an addition is accepted.

The default shortlist has at most two ranked seeds. Whole uncovered mirror groups
normally must fit a three-new-rib cap; selection stops once at least two ribs enter.
Under both mirror axes, the first shortlisted seed may instead add exactly four
new ribs as a whole group, then selection ends. Later seeds retain the ordinary
cap even if the first seed added nothing. This exception is used in the engineering
comparison; Examples I–V have at most one mirror axis.

Rationalization minimizes a smooth thickness-threshold count, followed by discrete
removal, joint reoptimization and recovery under a pass-specific compliance limit.
It is not direct active-set cardinality minimization, and is skipped for entering
layouts with at most three ribs. Every available member of a prescribed mirror
group must qualify for removal. Continuous convergence uses the stated numerical
feasibility tolerance; final reduced-layout acceptance uses the exact pass limit.

`tools/run_revision_experiments.py` provides insertion ablations, direct-deletion
controls, parameter sweeps and regular-layout checks. It defaults to a read-only
plan; execution needs `--execute`, a fresh `--output` and an explicit trial-count
guard. See [the revision guide](tools/REVISION_EXPERIMENTS.md). These controls show
the reported count/compliance/cost trade-offs, not universal optimizer superiority.

## Data and reproducibility

The fixed release is [v1.6.0](https://github.com/ZhifanLuo/rib-layout-optimization/releases/tag/v1.6.0).
Download its `rib-layout-optimization-v1.6.0-revision-data.zip` asset as well as the
source. The asset provides inputs/results, verification records, the table/figure
evidence index, current Figure 8 source and external-method material. Follow its
README for extraction layout, commands and the frozen engineering supplement.

Historical results retain their actual execution dates, configurations and source
fingerprints. Packaging a release does not rerun optimization; later source hashes
must not replace earlier ones in historical manifests. Rerun current inputs into
new directories and compare with the saved results; numerical trajectories can
vary across environments.

Repository `verification/` and `verification_v1_5_0/` remain historical records.
They do not override the current paper's release-data index. `configs/examples.yaml`
is archival. Executable case definitions and explicit paper JSON are inputs;
`configs/reported_results.json` supplies comparison targets, never optimizer input.

The external Amir–Majdouba plate–beam workflow has different objectives, support
and junction conventions and volume accounting. The data asset retains its input
transformations, saved state and native verification. Its historical launcher
requires GNU Octave plus local executable/output paths; Python requirements do
not install Octave. The comparison concerns engineering outcomes, not a direct
accuracy or optimality ranking.

## License and citation

This repository's Python implementation is [BSD-3-Clause](LICENSE). Third-party
code/data in the separate asset retain their own licenses and attribution;
external author code and separately sourced MMA routines are not relicensed here.
Consult the asset's license files.

Use `CITATION.cff` for software citation and cite the associated paper. v1.5.0 and
earlier tags remain available for historical results. No v1.6.0 DOI is claimed here
unless a separate archive subsequently assigns one.
