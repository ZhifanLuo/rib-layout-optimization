# Numerical verification package

This directory contains compact, machine-readable extracts from the
provenance-checked evidence run `20260804-085055`. It is intended to accompany
the Python source without duplicating bulky optimization histories, rib-design
payloads, or console logs.

## Frozen provenance

- Core executable/test aggregate SHA-256 (31 files):
  `9283b8f65554f017e20f7302439a41c1be58f980b7e834523237ea4109663bc6`.
- Diagnostic-tools aggregate SHA-256 (10 files):
  `7ed0e2d02a2245b117eabce1cf570d4478a32d474ff3dd18dbedeecdc95d7fb4`.
- Git commit present when the diagnostics ran:
  `276bd95b27bb01541d42e2f8ef9a058bffd10d6a`.
- Formal command: `cmd.exe /d /c "call example_all.bat <nul"`, executed
  sequentially from `Code/` with no quick-mode flag.
- Formal UTC interval: `2026-08-04T02:14:01.5844110Z` to
  `2026-08-04T02:34:20.8879100Z`; exit code `0`.
- Diagnostic matrix UTC interval: `2026-08-04T02:39:12.8195414Z` to
  `2026-08-04T03:21:02.0976691Z`; every command exited `0`.

The five diagnostic entry tools were frozen at these SHA-256 values:

| Tool | SHA-256 |
| --- | --- |
| `tools/run_sensitivity_verification.py` | `5837758258aa8a420dd7ffd525e2f99056097d7ed1e02448089bbb88457655e3` |
| `tools/run_robustness_study.py` | `809c62a75af23da772d6a82bb66e446336b5416d0195575dda15eb649e4d85b3` |
| `tools/run_geometry_restart_diagnostic.py` | `26d2c8d705cff6ab6abee00d1a79df49c0c6c9a4ebba6d8a67e991838f9d69f8` |
| `tools/run_full_pool_baseline.py` | `6c415cdae073bdb376bbe25c39b3fa3418abcea6a11f6dd41b811c5159c143b5` |
| `tools/run_topology_lifting_diagnostic.py` | `1bf2104b9f27c00ff2031cc72583bfbb034f78c18482e076d1f80f6750a53b13` |

The platform was Microsoft Windows 11 Home Chinese build `10.0.26200`,
64-bit, on an Intel Core Ultra 7 258V (8 physical/logical cores) with
33,101,520 KiB installed memory. The environment used Python 3.12.10,
NumPy 2.5.1, SciPy 1.18.0, single-threaded solver/sensitivity settings, and
`OMP_NUM_THREADS=4` in the shell environment. Elapsed times are observations
on that platform, not portable complexity measures.

## Package contents and source fields

| File | Evidence source and fields |
| --- | --- |
| `formal_stage_summary.csv` | `Code/results/example_N/results.json:mesh_used,quick,stages[*].{name,rib_count,compliance,volume,analyses}` |
| `sensitivity_by_step.json` | `sensitivity_verification.json:summary`, grouped by variable space, derivative, and step; trace-changing samples remain separate |
| `robustness_summary.csv` | `robustness_study.json:runs[*]`, including all 18 preregistered runs and common-mesh response |
| `fixed_layout_mesh_response.csv` | `robustness_study.json:fixed_layout_mesh_reanalysis.meshes` plus `response_160x80.json:response` |
| `restart_group_summary.json` | `restart_matrix_summary.json`, including all 13 groups, 45 runs, process totals, feasibility totals, and termination totals |
| `full_pool_stage_summary.csv` | `full_pool_case_1.json:stages[*]` and `full_pool_case_2.json:stages[*]`, with incremental and derived cumulative FE counts |
| `lifting_summary.csv` | Case-III/IV `topology_lifting_results.json` saved/reanalysed/lifted response and termination fields |
| `run_response_only_160x80.py` | Exact auxiliary script used for the fourth point in the fixed-layout response-only series |
| `manifest.json` | Machine-readable provenance, derivation, source-field, and file-hash mapping |
| `SHA256SUMS` | SHA-256 checksums for all package files except `SHA256SUMS` itself |

The `source` columns intentionally identify the original run artifacts. Those
raw artifacts are not copied here; a durable archival release should retain
them separately if full iteration-level replication is required.

## Reproduction commands

Run all commands from `Code/` in Windows PowerShell after creating the Python
environment described in the main README. Reproduce the four formal cases
first:

```powershell
cmd.exe /d /c "call example_all.bat <nul"
```

The exact sensitivity, robustness, full-pool, and topology-lifting commands
are given in the main `README.md`. The restart/multistart matrix used the
common options `--restarts 1 --thickness-perturbation 0.10
--endpoint-perturbation 0.05` and the following frozen groups:

| Output group | Case | Stage | Move step | Multistarts | Seed |
| --- | ---: | --- | ---: | ---: | ---: |
| `case1_geometry_move050` | 1 | geometry | 0.50 | 2 | 202608041 |
| `case2_geometry_move005` | 2 | geometry | 0.05 | 3 | 202608042 |
| `case2_geometry_move025` | 2 | geometry | 0.25 | 3 | 202608043 |
| `case2_geometry_move050` | 2 | geometry | 0.50 | 3 | 202608044 |
| `case2_rationalized_move005` | 2 | rationalized | 0.05 | 3 | 202608045 |
| `case2_rationalized_move025` | 2 | rationalized | 0.25 | 3 | 202608046 |
| `case2_rationalized_move050` | 2 | rationalized | 0.50 | 3 | 202608047 |
| `case3_geometry_move050` | 3 | geometry | 0.50 | 2 | 202608048 |
| `case3_rationalized_move050` | 3 | rationalized | 0.50 | 2 | 202608049 |
| `case4_geometry_move005` | 4 | geometry | 0.05 | 2 | 202608050 |
| `case4_geometry_move050` | 4 | geometry | 0.50 | 2 | 202608051 |
| `case4_rationalized_move005` | 4 | rationalized | 0.05 | 2 | 202608052 |
| `case4_rationalized_move050` | 4 | rationalized | 0.50 | 2 | 202608053 |

For each row, substitute its values into:

```powershell
.\.venv\Scripts\python.exe tools\run_geometry_restart_diagnostic.py --case CASE --source results\example_CASE\results.json --output diagnostics\restarts\OUTPUT_GROUP --stage STAGE --restarts 1 --multistarts MULTISTARTS --thickness-perturbation 0.10 --endpoint-perturbation 0.05 --seed SEED --initial-move-step MOVE_STEP
```

After the robustness command has produced
`diagnostics/robustness_case2/robustness_study.json`, reproduce the auxiliary
fixed-layout `160 x 80` response point with exactly one FEA:

```powershell
.\.venv\Scripts\python.exe verification\run_response_only_160x80.py --project-root .. --source diagnostics\robustness_case2\robustness_study.json --output diagnostics\response_only_160x80\response_160x80.json
```

The auxiliary script is post-processing support. It is not imported by the
formal solver and is not solver input.

## Status and convergence semantics

`process_status=complete` means that the Python process completed and emitted
its diagnostic payload. It is deliberately separate from numerical
convergence. Only a phase termination field equal to `converged` is treated as
convergence. A phase marked `true_response_backtracking_failed` returned its
best volume-feasible incumbent after the bounded true-response safeguard
failed to accept a new step; it must not be relabelled as converged.

All 45 restart processes completed and were volume-feasible, but only 18
terminated `converged`; 27 terminated
`true_response_backtracking_failed`. Likewise, the `80 x 40` mesh robustness
run and the 2.5% relaxation run completed as processes but did not converge in
their last geometry phases. Adverse runs are part of the evidence set.

## Evidence boundary and known limitations

- Sensitivity results support smooth, fixed-trace derivatives. Endpoint
  perturbations that alter the discrete rib-to-mesh trace are explicitly
  classified as non-smooth and are not used as classical derivative evidence.
- The fixed-layout `20 x 10` to `160 x 80` response sequence has decreasing
  successive changes but has not plateaued; it does not establish mesh
  convergence. End-to-end mesh runs additionally change the short-rib filter
  and topology.
- The robustness and restart matrices quantify Case-II parameter and local-path
  dependence. They are not global-optimality evidence.
- The full-pool calculation is an internal Pareto comparison using the same
  implementation, not an independent MMC, continuum-topology, or
  element-connectivity benchmark. The dense pool retains many more ribs and
  therefore does not represent equal design-space work.
- The topology-lifting runs are consistency checks, not local-optimality
  proofs.
- Connected rib intersections and Boolean-union volume accounting, a curved
  multi-load shell with stress/buckling/frequency/manufacturing constraints,
  and an independently implemented external baseline remain outside the
  present evidence.

## Release state

This package is prepared for software version `v1.1.0`. The stable Zenodo
concept DOI is `10.5281/zenodo.21638172`; the immutable DOI for `v1.1.0` is
pending publication of the corresponding Zenodo archive and must not be
invented or replaced by the concept DOI. The preceding immutable `v1.0.1`
archive (`10.5281/zenodo.21782271`) does not contain this verification package.
