# Rib layout optimization — reference Python implementation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21638172.svg)](https://doi.org/10.5281/zenodo.21638172)

This folder contains the executable reference implementation of the
multi-phase method described in *An Active-Set Framework for Explicit
Stiffening-Rib Layout Optimization*. It implements:

1. compliance minimization with a rib-volume and thickness bounds;
2. performance-validated thickness filtering using only the prescribed
   thickness threshold and compliance check;
3. direct candidate ranking by frozen-field stiffness energy per added rib
   volume;
4. fixed-connectivity endpoint geometry optimization;
5. smooth-Heaviside rationalization, threshold deletion, and geometry
   optimization of the reduced active set.

## Implementation scope

This repository is the executable Python reference implementation of the
paper's abstract shell-based model. The case-specific numerical choices are
explicit in `example1.py` through `example4.py`, while the finite-element code
uses its own documented MITC4 shell and tied-interface formulation. Results
should therefore be reproduced with the supplied Python entry points and a
versioned environment rather than interpreted as cross-solver equivalence.

The former combined
`configs/examples.yaml` file is retained only as an archival reference. The
wall is represented by six-DOF four-node MITC4 Reissner--Mindlin flat-shell
elements. MITC4 assumed-natural shear removes the hourglass modes of one-point
integration without introducing thin-shell shear locking. Drilling rotations
are stabilized consistently against the continuum in-plane rotation; the
element has exactly six rigid-body modes.

Each rib is a vertical MITC4 shell strip. Its free top-edge degrees of freedom
are retained in the global sparse FEA system; this is algebraically equivalent
to static condensation but avoids the dense ground-node Schur complement of a
long rib. The bottom edge is split at every ground-grid
intersection and refined four times per interval; all six bottom-edge degrees
of freedom are bilinearly mapped to the ground shell. The Example-I interface
interpolation mismatch is checked after every formal rerun; in the current
result it is `0.0034%` for translations and `0.55%` for rotations. Rib
intersections are intentionally not tied to each other. The reported values are kept separately in
`configs/reported_results.json`; they are used only for comparison and are
never injected into the optimizer.

The active-set loop first repeats sizing/filtering until a complete pass deletes
no rib. Throughout one repeated-filtering phase, every deletion is checked
against the compliance fixed at the start of that phase, with at most `1%`
degradation allowed. A rib also enters the deletion batch when its length is
below `max(3*min(dx,dy), 0.25*initial_rib_cell_size)` and its thickness is below
five times the current thickness threshold. Thin and short/light deletions use
the same resizing/FEA performance validation. It then adds a limited candidate batch and performs sizing
before any filtering. If this sizing-only design improves compliance by less
than `1%`, the complete addition is reverted and the active-set loop terminates.
Otherwise the addition is accepted and sizing/filtering is repeated to deletion
convergence. After this complete addition/filtering cycle, a net compliance
improvement below `1%` terminates the active-set loop and starts geometry
optimization; otherwise the next member-addition round begins. Tried candidates are not
immediately reinserted if filtering removes them. If a member-addition round
retains no new rib after filtering, adaptive optimization stops immediately
and geometry optimization begins; no further candidate is attempted. All
examples rank every candidate directly by its frozen-displacement-field
stiffness energy divided by the volume added at the prescribed initial
thickness. At most two independently ranked candidate seeds are then checked.
Under mirror symmetry,
if the first seed needs a new reflected partner, that two-rib orbit fills the
batch immediately. Only when the first seed needs no new partner is the second
seed considered; if the second seed needs a reflected partner, it is also
added, so a symmetry-completed batch contains at most three ribs. A candidate
wholly covered by an existing rib, or by the continuous union of several
existing collinear ribs, is
skipped. If one collinear candidate completely covers another, the covering
longer rib is always retained to favor manufacturability, regardless of
ranking-score difference. Partial overlaps retain the higher-ranked
candidate; equal-length duplicates retain their ranking order. This length
preference does not apply to non-collinear candidates or to collinear
candidates that only meet at an endpoint. Every selected candidate must have
a stiffness-per-added-volume factor at least `70%` of the maximum factor among
eligible candidates. Candidates fully covered by one existing
rib or a gap-free union of existing collinear ribs are excluded before this
reference is calculated; exactly `70%` is eligible. Covered or overlapping candidates are skipped while the
ranked pool is scanned for a replacement. Member addition terminates only
when the pool contains no valid uncovered candidate. If a selected longer
candidate wholly covers an existing shorter rib, the trial active set replaces
the shorter rib with the longer candidate before sizing. A candidate wholly
covered by one existing rib or by a gap-free union of existing collinear ribs
is still skipped. Partial collinear overlap is left unchanged for now.
These geometric coverage rules are applied to the single direct
stiffness-per-volume ranking path.
Geometry optimization treats every active thickness and endpoint coordinate
simultaneously. Endpoint coordinates are bounded only by the ground-shell
domain; their local bounds are recomputed around the current coordinates after
every FEA by the enhanced-MMA move-limit rule. Rationalization uses the same
dynamic move limits. It solves Eq. (18) once with `tref` equal to the
`q = 1/n_rib + rho` thickness quantile of the geometry-stage ribs, where
`n_rib` is the number of ribs immediately before rationalization and `rho` is
the prescribed compliance relaxation. This value is calculated once and kept
fixed throughout rationalization. Its projection continuation starts at
`beta=1`, increases beta by one
per outer iteration, and caps it at `beta=10`; convergence checks start only at
the cap, so the two-consecutive-step rule requires at least eleven iterations.
True Eq. (18) compliance up to `1.001*Cref` is treated as feasible and enters
the convergence test. If no thickness falls below `tref`, rationalization
restores the preceding geometry result and stops. Otherwise it deletes the
complete thin/short-light filtering batch and solves Eq. (7). The final
acceptance limit is exactly `Cref`. If the result violates `Cref`, let `n_rrib`
be the number of ribs deleted in the initial discrete deletion attempt. Each
recovery round restores up to the fixed number
`max(1, ceil(n_rrib/3))` of ribs that remain deleted, ranked by their saved
Eq. (18) thickness in descending order with original rib index breaking ties.
The restored seeds are completed to their available mirror groups only when
mirror symmetry is configured; otherwise the rib decisions are independent.
The seed target remains fixed across repeated recovery rounds. This continues
until the limit is met or no deletion can be accepted. Eq. (18)
and these post-deletion Eq. (7) solves use `Gstep=0.5`. If none is accepted, the preceding geometry
result is restored. No standalone deletion-verification FEA is performed; the
reduced topology enters Eq. (7) directly, and its necessary initialization FEA
is counted as part of Eq. (7). The former threshold homotopy is not used.

For endpoint coordinates, the move-limit half-width is
`Gstep * Lstep_i * 2*dx` in x and `Gstep * Lstep_i * 2*dy` in y. With the
default initial `Gstep=0.5` and `Lstep_i=1`, this gives initial half-widths
`dx` and `dy`. Thickness variables retain the enhanced-MMA current-value/range
scale. A successful same-direction history multiplies the variable factor by
`1.2`; oscillation multiplies it by `0.7`. Direction detection uses moves
normalized by each variable's global range; if either consecutive normalized
move is within `1e-6` of zero, no `1.2/0.7` factor is applied. For geometry
optimization, each convex trial is checked by a true FEA before it becomes the
next outer design. A feasible trial whose relative compliance increase exceeds
`1e-4` (0.01%) is rejected; the global move limit is contracted by the existing
unsuccessful-step factor `0.75`, and the same approximation is resolved. At
most four contracted retries are permitted after the first trial. Exhausting
this bound terminates the geometry stage with
`true_response_backtracking_failed`; this safeguard parameter is not a
convergence criterion. Sizing and geometry optimization retain and return the
lowest-compliance feasible true-FEA incumbent encountered, including rollback
from an inferior last iterate.

The main geometry stage and rationalization have distinct initial-move
settings: `geometry_move_limit_initial` and
`rationalization_move_limit_initial`. Both default to `0.50`. Example II
overrides both values to `0.05`; Examples I, III, and IV retain `0.50`. This is
a case-specific calibration based on saved adaptive-stage restart diagnostics:
the `case2_adaptive_move_005` run converged to
`C = 13.8187025742`, whereas the tested `0.50` and `0.10` starts terminated by
the true-response backtracking safeguard. This limited diagnostic observation
does not establish robustness or global convergence.
There is no fixed cumulative endpoint-displacement interval; only the ground-
shell domain clips the adaptive local bounds.
If an inner convex solution makes a rib newly too short or creates a new
collinear overlap, only the responsible rib endpoint coordinates are fixed at
their current-outer-iterate positions and the same approximation is resolved.
The other rib coordinates and all thicknesses remain active. This local
geometry freeze does not contract the global `Gstep` and does not call FEA.

Sizing, geometry optimization, and rationalization use sequential convex
approximation (SCA). Every outer iteration follows the same sequence:

1. run one FEA at the current design and compute sensitivities;
2. construct reciprocal/affine convex approximations with proximal terms and
   a linearized volume constraint;
3. solve that approximate subproblem without any FEA inside the optimizer;
4. evaluate the inner optimum using true FEA;
5. for geometry optimization, reject a materially worse feasible response,
   contract the move limit, and resolve the same approximation within the
   bounded retry safeguard; otherwise accept it as the next outer design;
6. declare convergence only when the true constraints are satisfied within
   `0.1%`, the normalized design-variable change is below `1.0%`, and either
   the objective change is below `0.5%` or normalized design-variable change
   is below `0.1%`, for two consecutive outer steps.

A geometry stage terminates by one of three explicitly reported conditions:
the convergence declaration above, the configured outer-iteration limit, or
failure to obtain an acceptable true response within the bounded backtracking
retries. Only the first condition is convergence.

The Eq. (9) sizing problem is implemented as the thickness-only specialization
of the Eq. (7) separable convex subproblem. Both use the same closed-form box-
constrained update and one-dimensional bisection on the volume multiplier; no
SLSQP or FEA is used inside either convex subproblem. The multiplier is searched
in both directions: a volume-slack unconstrained solution is not returned
immediately. The solver targets the active volume bound when it is reachable
inside the move box and otherwise returns the box solution with the smallest
absolute linearized-volume residual.

The global FEA uses Intel MKL Pardiso when `pypardiso` is installed and checks
the linear residual after every solve. Formal runs pin Pardiso/OpenMP to one
thread and disable dynamic MKL threading so repeated optimization paths use a
deterministic floating-point reduction order. The runtime rejects any shell
model requesting `linear_solver_threads != 1` or `sensitivity_workers != 1`,
so formal and diagnostic entry points cannot silently re-enable parallelism.
SciPy SuperLU symmetric mode is
the automatic fallback. Because every optimization FEA changes the stiffness
matrix, the retained Pardiso factorization is explicitly released after each
residual-checked solve; this prevents MKL memory growth over long runs without
changing the response. Rib sensitivity uses an equivalent condensed-energy
formulation and the envelope theorem, so each rib top field is equilibrated
once per outer iteration instead of once for every coordinate perturbation.
Each convex Eq. (18) rationalization approximation is solved through its
two-variable compliance/volume dual. For fixed multipliers, rib coordinates
have an explicit minimizer and each thickness is the unique positive root of
its monotone cubic KKT equation. This removes the former high-dimensional
SLSQP iteration; its normalized constraint tolerance is configured as
`rationalization_dual_tolerance: 1.0e-9`.

No contribution-based filtering is used. Active-set filtering removes only
ribs below `10^-r t_0`, followed by the manuscript compliance validation.
Example I starts at the enlarged threshold `0.1 t_0 = 0.02 mm`; the other
examples retain the `0.01 t_0` starting threshold.

Mirror symmetry is declared per case with `mirror_symmetry`. Axis `x` reflects
coordinates about `x=width/2`; axis `y` reflects them about `y=height/2`.
Initial and candidate ground structures must be closed under every configured
reflection. Member addition completes a strong candidate with its mirror
orbit. Filtering and rationalization delete an orbit only when every member
qualifies, while a failed rationalization trial completes each selected
recovery seed to its mirror orbit. This conservative group rule
preserves manufacturable symmetry under small load or numerical asymmetries.
Sizing, geometry optimization, and the rationalization Eq. (18) also assemble
their design vectors in reduced mirror variables: all ribs in one mirror orbit
share one thickness variable, and mirrored endpoint coordinates are represented
by exact affine relations. A self-symmetric rib keeps only its independent
endpoint coordinates. Full rib thicknesses and coordinates are expanded only
for FEA and output, so symmetry is exact while the continuous subproblems use
fewer design variables.

The initial structure follows Table 1: both diagonals of every square lattice
cell are active. Examples I and II use `10 mm x 10 mm` cells, `2 mm` rib
height, and initial rib length `10*sqrt(2) mm`. Examples III and IV use a
`200 mm x 100 mm` ground plane divided into `25 mm x 25 mm` cells, `10 mm`
rib height, and initial rib length `25*sqrt(2) mm`. All initial ribs are at
`+45 deg` or `-45 deg`; the four cases contain 8, 16, 64, and 64 ribs.
The rib-thickness upper bounds are `3 mm` for Examples III and IV. All four
examples perform rationalization with `5%` compliance relaxation. Examples III
and IV additionally execute the configured repeated pass, again with a `5%`
pass-specific relaxation; no `2%` or `10%` paths are configured.

## Requirements and installation

Python 3.12 is recommended; the publication checks were run with Python
3.12.10. The direct dependencies are declared in `requirements.txt`. The
tested local environment used NumPy 2.5.1, SciPy 1.18.0, Matplotlib 3.11.1,
PyYAML 6.0.3, and pypardiso 0.4.7. The requirement ranges permit later
compatible releases, so record `python --version` and `python -m pip freeze`
when creating a new archival result set.

Run the following commands from this `Code/` directory.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux or macOS shell:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, `setup_env.bat` performs the same basic environment creation and
dependency installation. The numerical implementation automatically uses
single-threaded MKL Pardiso when available and otherwise falls back to SciPy
SuperLU; `results.json` records the configured solver and thread settings.

## Reproduce the four examples

The publication entry point is the cross-platform `run_all.py`. With no
options it launches `example1.py` through `example4.py` in order using the
same Python interpreter, uses the configured full meshes, and stops at the
first failure. This is the Python equivalent of the retained Windows-only
`example_all.bat`.

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe run_all.py
```

The exact Windows command used for the provenance-checked publication run was:

```powershell
cmd.exe /d /c "call example_all.bat <nul"
```

Run it from `Code/`. It executes all four full cases sequentially and stops at
the first nonzero exit code.

Linux or macOS shell:

```bash
.venv/bin/python run_all.py
```

A single formal case can be run directly, for example:

```powershell
.\.venv\Scripts\python.exe example3.py
```

The default configuration is the only mode intended to reproduce the
reported case settings. Examples I and II use meshes `20 x 20` and `40 x 20`;
Examples III and IV use `80 x 40`. Examples III and IV also execute their
configured second rationalization pass. All cases run sequentially because
they share the `results/` output root.

## Quick and diagnostic execution

`--quick` is a diagnostic mode, not a source of publication values. It leaves
Examples I and II unchanged because those cases define no quick mesh, while
Examples III and IV use a `32 x 16` mesh and four shell elements along each
rib instead of their formal discretizations:

```powershell
.\.venv\Scripts\python.exe run_all.py --quick
```

`--geometry-sweeps N` overrides the configured maximum number of geometry
iterations and therefore changes the algorithmic run. For example,
`--geometry-sweeps 0` is only a fast workflow check. `--output PATH` redirects
all generated case folders. These options may be combined:

```powershell
.\.venv\Scripts\python.exe run_all.py --quick --geometry-sweeps 0 --output diagnostic_results
```

The utilities under `tools/` provide restart audits, rationalization
diagnostics, aggregate summaries, and post-processing. They are supporting
diagnostics rather than the formal four-case entry point.

### Numerical-verification commands

The compact reviewed tables from the August 2026 evidence run are under
`verification/`; see `verification/README.md` for provenance, field mappings,
and interpretation limits. The following commands reproduce the main
diagnostics from fresh formal `results/example_N/results.json` files.

Case-II thickness and endpoint sensitivity verification at three perturbation
steps, in both production mirror-reduced and unreduced spaces:

```powershell
.\.venv\Scripts\python.exe tools\run_sensitivity_verification.py --case 2 --source results\example_2\results.json --output diagnostics\sensitivity_case2_geometry --stage geometry --verification-space both --thickness-steps 0.01 0.003 0.001 --endpoint-steps 0.01 0.003 0.001
```

Case-II 18-run one-factor study (mesh, pool span, 70% retention threshold,
1% convergence threshold, 5% rationalization relaxation, and starting
layout), with a common `80 x 40` response reanalysis:

```powershell
.\.venv\Scripts\python.exe tools\run_robustness_study.py --case 2 --output diagnostics\robustness_case2 --studies mesh pool retention convergence relaxation starts --mesh-values 20x10 40x20 80x40 --pool-spans 1 2 3 --retention-thresholds 0.5 0.7 0.9 --convergence-thresholds 0.005 0.01 0.02 --relaxations 0.025 0.05 0.075 --starting-layouts all_orbits even_orbits odd_orbits --common-reanalysis-mesh 80 40 --seed 20260804
```

A deterministic restart/multistart group is run as follows; the reviewed
13-group matrix, including the exact case/stage/move-step/seed combinations,
is listed in `verification/README.md`.

```powershell
.\.venv\Scripts\python.exe tools\run_geometry_restart_diagnostic.py --case 2 --source results\example_2\results.json --output diagnostics\restarts\case2_geometry_move005 --stage geometry --restarts 1 --multistarts 3 --thickness-perturbation 0.10 --endpoint-perturbation 0.05 --seed 202608042 --initial-move-step 0.05
```

The full enumerated candidate pools for Cases I and II can be carried through
sizing, geometry optimization, and one 5% rationalization pass:

```powershell
.\.venv\Scripts\python.exe tools\run_full_pool_baseline.py --case 1 2 --output diagnostics\full_pool_cases1_2 --post-sizing-policy rationalization --rationalization-relaxation 0.05
```

The topology-lifting consistency checks are:

```powershell
.\.venv\Scripts\python.exe tools\run_topology_lifting_diagnostic.py --case 3 --source results\example_3\results.json --output diagnostics\lifting\case3_geometry_to_rationalized --full-stage geometry --reduced-stage rationalized
.\.venv\Scripts\python.exe tools\run_topology_lifting_diagnostic.py --case 4 --source results\example_4\results.json --output diagnostics\lifting\case4_geometry_to_rationalized --full-stage geometry --reduced-stage rationalized
```

`process_status=complete` means that a diagnostic command finished and wrote
its payload. It does **not** mean that every optimization phase converged.
Only `termination_reason=converged` (or the corresponding phase field) is a
convergence declaration; `true_response_backtracking_failed` means that the
best feasible incumbent was returned after the bounded safeguard stopped the
phase. Preserve unsuccessful and adverse numerical outcomes when reporting a
matrix.

The convergence and comparison diagnostics accept both module and direct
script invocation. For example, these two forms are equivalent:

```powershell
.\.venv\Scripts\python.exe -m tools.run_geometry_restart_diagnostic --case 3 --source results\example_3\results.json --output diagnostics\restart3 --restarts 2 --multistarts 4
.\.venv\Scripts\python.exe tools\run_geometry_restart_diagnostic.py --case 3 --source results\example_3\results.json --output diagnostics\restart3 --restarts 2 --multistarts 4
```

Multistarts can apply seeded, symmetry-preserving thickness and endpoint
perturbations; they do not constitute a global-optimality proof. A
topology-lifting consistency check
can reinsert ribs deleted between two saved stages:

```powershell
.\.venv\Scripts\python.exe -m tools.run_topology_lifting_diagnostic --case 4 --source results\example_4\results.json --output diagnostics\lifting4 --full-stage geometry --reduced-stage rationalized
```

Lifting requires compatible source data. The tool checks case, quick-mode and
mesh metadata, then reanalyzes both saved stages with the current executable;
it stops if either compliance differs from the saved value by more than the
configurable `--source-compliance-tolerance` (default `1e-6` relative).

The complete generated candidate pool can be optimized without active-set
screening as an internal baseline:

```powershell
.\.venv\Scripts\python.exe tools\run_full_pool_baseline.py --case 1 2 --output diagnostics\full_pool --post-sizing-policy rationalization --rationalization-relaxation 0.05
```

This full-pool calculation uses the same solver and candidate restrictions as
the proposed method. It is an internal algorithmic baseline, not an independent
literature-method comparison, and large formal pools can require substantial
memory and runtime. None of these tools writes publication `results/` unless
that location is explicitly passed through `--output`.

## Outputs

Each case writes to `results/example_N/` by default. The common artifacts are:

- `checkpoint_results.json`: numerical state saved before optional plotting;
- `results.json`: mesh/mode metadata, stage rib endpoints and thicknesses,
  compliance, volume, analysis counts, histories, and elapsed time;
- `summary.csv` and `case_summary.csv`: compact stage-level statistics;
- `all_stages.png`, `active_set_iterations.png`, and one PNG for each stored
  optimization stage;
- rationalization history JSON and thickness CSV files when the corresponding
  rationalization iterations occur.

Output directories are regenerated data and are ignored by default. Compact
reviewed evidence tables are retained under `verification/`; full raw outputs
should be attached to an archival release when their size is appropriate.

## Tests

Run the complete regression suite from `Code/`:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

On Linux or macOS, replace the interpreter path with `.venv/bin/python`.
Unit tests exercise components and orchestration but do not replace a complete
four-case run when validating publication results.

## Citation and license

Software citation metadata for the prepared `v1.1.0` release are provided in
`CITATION.cff`, and the development repository is available at
<https://github.com/ZhifanLuo/rib-layout-optimization>. GitHub exposes these
metadata through its **Cite this repository** interface. The stable Zenodo
concept DOI, <https://doi.org/10.5281/zenodo.21638172>, resolves the complete
release family. The immutable archive for `v1.1.0` is available at
<https://doi.org/10.5281/zenodo.21784031>. For release history, the preceding
immutable `v1.0.1` archive is available at
<https://doi.org/10.5281/zenodo.21782271>. ORCID and article publication
metadata remain to be inserted when available.

The implementation is distributed under the BSD 3-Clause License; see
`LICENSE`.

## File map

The principal user-facing Python files are in `Code/`:

- `example1.py` through `example4.py` — FE dimensions, mesh, material, loads,
  supports, rib limits, case-level algorithm overrides, and custom outputs;
- `rib_layout_env.py` — deterministic runtime, solver-thread, and path setup;
- `rib_layout_core.py` — shared configuration, model construction, and the
  common sizing/filtering/adaptive/geometry/rationalization workflow;
- `rib_layout_output.py` — common JSON, CSV, history, and plot output manager.

The tested numerical implementation remains separated under
`rib_layout_algorithms/`; ordinary users do not need to edit these files:

- `rib_layout_algorithms/shell.py` - six-DOF MITC4 Reissner--Mindlin flat-shell element;
- `rib_layout_algorithms/model_shell.py` - sparse tied ground/rib assembly, FE solve,
  condensed-energy thickness/coordinate sensitivities, and solver fallback;
- `rib_layout_algorithms/move_limit.py` - enhanced-MMA move-limit state;
- `rib_layout_algorithms/optimization.py` - sizing, filtering, member addition, geometry, and
  rationalization algorithms;
- `rib_layout_algorithms/plotting.py` - shared stage visualizations;
- `rib_layout_algorithms/frame.py` and `rib_layout_algorithms/model.py` - retained validated reference model;
- `run_all.py` - cross-platform sequential runner for all four cases;
- `tools/` - combined compatibility runner, diagnostics, and
  stage-figure generation;
- `tests/` - complete regression tests;
- `configs/` - archival case configuration and manuscript comparison data;
- `requirements.txt` and `setup_env.bat` - reproducible setup.

## Units

The implementation uses N and mm. The aluminium-alloy modulus `E = 70 GPa` is
therefore entered as `70,000 N/mm²`; compliance has units N·mm.
