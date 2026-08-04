# Numerical verification package

This directory contains compact, machine-readable extracts from the
provenance-checked evidence run `20260804-153636`. It accompanies the Python
source without duplicating the large iteration histories under `diagnostics/`.

## Frozen provenance

- Core executable/test aggregate SHA-256 (31 files):
  `35adae362d057c25732ccda057fe06c0be84c18beef328bac659ecdaf2017d6b`.
- Executable-plus-tools aggregate SHA-256 (41 files):
  `bd956744f58cd0bfc74713947dc1441b5c005240addcaf5d82937f8ee33c4856`.
- Diagnostic-tools aggregate SHA-256 (10 files):
  `0e1b0775d4daff53607b922978f774204ab00654aaacdefcce5044e542f78d89`.
- Git commit present while the working-tree diagnostics ran:
  `a2bdfe80e8a5563723d72bef887c1e4115ec8666`. The working tree was dirty;
  the aggregate hashes above, rather than the commit alone, identify the
  executed source.
- Formal command: `cmd.exe /d /c "call example_all.bat <nul"`, executed
  sequentially from `Code/` with no quick-mode flag.
- Formal UTC interval: `2026-08-04T07:56:31.6572071Z` to
  `2026-08-04T08:15:34.8146633Z`; exit code `0`.
- Diagnostic UTC interval: `2026-08-04T08:22:41.6561783Z` to
  `2026-08-04T09:13:38.2330864Z`; every requested command exited `0`.

The diagnostic entry tools were frozen at these SHA-256 values:

| Tool | SHA-256 |
| --- | --- |
| `tools/run_sensitivity_verification.py` | `5837758258aa8a420dd7ffd525e2f99056097d7ed1e02448089bbb88457655e3` |
| `tools/run_robustness_study.py` | `809c62a75af23da772d6a82bb66e446336b5416d0195575dda15eb649e4d85b3` |
| `tools/run_geometry_restart_diagnostic.py` | `d6a3826a3f55a97c1c6c721a924d6d46fd8d7b85b678a48f2739478c55e646fc` |
| `tools/run_full_pool_baseline.py` | `6c415cdae073bdb376bbe25c39b3fa3418abcea6a11f6dd41b811c5159c143b5` |
| `tools/run_topology_lifting_diagnostic.py` | `1bf2104b9f27c00ff2031cc72583bfbb034f78c18482e076d1f80f6750a53b13` |
| `verification/run_response_only_160x80.py` | `f59ff2481298672db3f0a862fee899893a4a68784ad73396ec9014dcf11f00c7` |

The platform was Microsoft Windows 11 build `10.0.26200`, 64-bit, on an
Intel Core Ultra 7 258V (8 physical/logical cores) with 33,895,956,480 bytes
installed memory. The environment used Python 3.12.10, NumPy 2.5.1,
SciPy 1.18.0, single-threaded solver/sensitivity settings, and
`OMP_NUM_THREADS=4`. Elapsed times are platform observations, not portable
complexity measures.

## Package contents and source fields

| File | Evidence source and fields |
| --- | --- |
| `formal_stage_summary.csv` | `results/example_N/results.json:mesh_used,quick,stages[*].{name,rib_count,compliance,volume,analyses}` |
| `sensitivity_by_step.json` | `diagnostics/sensitivity_case2_geometry/sensitivity_verification.json:components`, grouped by variable space, derivative and step; trace-changing samples remain separate |
| `robustness_summary.csv` | `diagnostics/robustness_case2/robustness_study.json:runs[*]`, retaining all 18 registered runs and common-mesh responses |
| `fixed_layout_mesh_response.csv` | `robustness_study.json:fixed_layout_mesh_reanalysis.meshes` plus `response_160x80.json:response` |
| `restart_group_summary.json` | All 13 `diagnostics/restarts/*/geometry_restart_results.json` payloads, including all 45 runs and terminations |
| `full_pool_stage_summary.csv` | `full_pool_case_1.json:stages[*]` and `full_pool_case_2.json:stages[*]`, with cumulative analysis counts derived in stage order |
| `lifting_summary.csv` | Case-III/IV `topology_lifting_results.json` rib counts, saved/reanalysed/lifted responses, analysis counts and termination |
| `run_response_only_160x80.py` | Unchanged auxiliary script used for the fourth fixed-layout response point |
| `manifest.json` | Machine-readable provenance, derivation, source-field and file-hash mapping |
| `SHA256SUMS` | SHA-256 checksums for every package file except `SHA256SUMS` itself |

Raw diagnostic artifacts remain under `Code/diagnostics/`. The `source`
columns point to those current artifacts. A durable release should archive both
the compact package and the raw artifacts if iteration-level replication is
required.

## Reproduction commands

Run all commands from `Code/` in Windows PowerShell. Reproduce the four formal
cases first:

```powershell
cmd.exe /d /c "call example_all.bat <nul"
```

The exact sensitivity, robustness, full-pool and lifting commands are given in
the main `README.md`. The restart matrix used the common options
`--restarts 1 --thickness-perturbation 0.10 --endpoint-perturbation 0.05` and
these fixed groups:

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

For each row substitute its values into:

```powershell
.\.venv\Scripts\python.exe tools\run_geometry_restart_diagnostic.py --case CASE --source results\example_CASE\results.json --output diagnostics\restarts\OUTPUT_GROUP --stage STAGE --restarts 1 --multistarts MULTISTARTS --thickness-perturbation 0.10 --endpoint-perturbation 0.05 --seed SEED --initial-move-step MOVE_STEP
```

After the robustness command, reproduce the auxiliary response point with one
FEA and no optimization:

```powershell
.\.venv\Scripts\python.exe verification\run_response_only_160x80.py --project-root .. --source diagnostics\robustness_case2\robustness_study.json --output diagnostics\response_only_160x80\response_160x80.json
```

## Status and convergence semantics

`process_status=complete` means that a diagnostic process completed and wrote
its payload. It remains distinct from numerical convergence. For this evidence
run all 18 robustness processes and all 45 restart processes completed,
remained volume-feasible, and reported final geometry termination
`converged`. The compact files retain every run, including higher-compliance
local paths; convergence does not imply a global optimum.

No `objective_rollback_failed_*` termination occurred in the current matrices.
This absence shows only that the 50% safeguard did not exhaust its bounded
retry mechanism on these paths; it does not prove that the safeguard is
inactive for all problems.

The raw `robustness_study.json` uses Python's permissive JSON encoding for two
undefined `approximation_ratio` values in the even-orbits rationalization
iteration history (`runs[16]`, records 31 and 32), so those two values appear
as non-standard `NaN`. They are undefined diagnostic ratios, not compliance,
volume, topology, termination or convergence values. The compact package JSON
files are strict JSON and the robustness CSV omits this undefined internal
ratio. This serialization limitation is retained explicitly for provenance.

## Evidence boundary and known limitations

- Sensitivity results support smooth, fixed-trace derivatives. At step 0.001,
  reduced-space thickness and endpoint maximum relative errors are
  `3.3175666875238265e-6` and `0.003460923676189275`. At step 0.01, however,
  the reduced endpoint maximum error rises to `0.2863528319801101`; four
  full-space endpoint samples also change trace. The larger-step adverse result
  is retained rather than used as classical derivative evidence.
- The fixed five-rib response sequence is 12.43274788223292,
  13.08414981614593, 13.488160942389507 and 13.981085103377055 on meshes from
  20 x 10 through 160 x 80. It has not plateaued and does not establish mesh
  convergence. End-to-end mesh runs additionally change topology and filtering.
- The robustness and restart matrices quantify parameter and local-path
  dependence. They are not global-optimality evidence. In particular, all 45
  restarts converged but their final compliance distributions remain distinct.
- The full-pool calculation is an internal Pareto comparison using the same
  implementation, not an independent MMC, continuum-topology or
  element-connectivity baseline. Case I retains 31 of 36 candidates after
  rationalization, and Case II retains all 86.
- The lifting runs are consistency/local-path diagnostics, not optimality
  proofs. Reoptimization after lifting improves Case III from
  17.501170545842463 to 16.45881398550952 and Case IV from
  266.3127460952643 to 265.598296539515, demonstrating remaining path
  dependence rather than an effect of adding zero-thickness ribs alone.
- Connected rib intersections and Boolean-union volume accounting, a curved
  multi-load shell with stress/buckling/frequency/manufacturing constraints,
  and an independently implemented external baseline remain outside the
  present evidence.

## Release state

The stable Zenodo concept DOI is `10.5281/zenodo.21638172`. The immutable
`v1.1.0` archive is `10.5281/zenodo.21784031`, but it predates this rerun and
does not contain the current source/results/package. A new versioned archive is
required before citing these artifacts as a released replication package.
