# v1.2.0 compact publication evidence

This directory contains the reviewed, compact evidence index for GitHub tag
`v1.2.0` and Zenodo version DOI
[`10.5281/zenodo.21820166`](https://doi.org/10.5281/zenodo.21820166).
The stable concept DOI is
[`10.5281/zenodo.21638172`](https://doi.org/10.5281/zenodo.21638172).

## Release archives

The release is distributed as three archives with identical assets on GitHub
and Zenodo:

1. `rib-layout-optimization-v1.2.0-source.zip` — frozen source and this compact
   index;
2. `rib-layout-optimization-v1.2.0-formal-results.zip` — complete formal outputs
   for Cases I–IV;
3. `rib-layout-optimization-v1.2.0-verification-results.zip` — complete
   sensitivity, one-factor, restart/multistart, fixed-layout mesh, and full-pool
   diagnostic outputs plus run manifests.

The frozen executable snapshot contains 41 source files with aggregate SHA-256
`d7a585a52198f6bff6cfd8096c8768d81c83349e5d5807a611fdf4385537a3c7`.
The four-case command ran sequentially and all 19 diagnostic commands exited
zero. All 16 formal JSON and all 84 diagnostic JSON files parse as strict JSON;
all fresh formal and diagnostic files are free of workstation-specific absolute
paths. Mathematically undefined `0/0` approximation ratios are represented as
`null` with explicit `undefined` status and
`predicted_objective_change_near_zero` reason.

## Current compact files

| File | Contents | Full release source |
| --- | --- | --- |
| `formal_stage_summary.csv` | All 18 stored stages for Cases I–IV | formal-results archive, `results/example_N/results.json` |
| `sensitivity_by_step.json` | Cases II–IV by-step smooth/nonsmooth error summaries | verification-results archive, `diagnostics/sensitivity_case*_geometry/` |
| `robustness_summary.csv` | All 18 registered Case-II one-factor runs | verification-results archive, `diagnostics/robustness_case2/` |
| `fixed_layout_mesh_response.csv` | One fixed five-rib layout at `20x10` through `160x80` | robustness and response-only diagnostics |
| `restart_group_summary.json` | All 13 groups and their 45-run distributions | verification-results archive, `diagnostics/restarts/` |
| `full_pool_stage_summary.csv` | Cases I and II complete-pool sizing, geometry, and rationalization | verification-results archive, `diagnostics/full_pool_cases1_2/` |
| `manifest.json` | Release identity, frozen-source/result hashes, and field-level provenance | both result archives |
| `SHA256SUMS` | Digests for this current compact set and the required response-only script | this directory |

The complete formal result hashes are:

- Case I: `ea6758c241ad2b642c92c162def0f71f025b1c876cee23e90f855ffe68a037bc`
- Case II: `27cbb67d2084f5b9e291001a993ca3950920e3eec049c795f630942c47e1d6fb`
- Case III: `6b51917d2acef88a7d96cf27b25c5fd58274a50961b8a65dd387d2e83df4ac4b`
- Case IV: `848e646291d9929d23da2dbeb8e15842db9f15e4d66bc3b732d966435f1007a6`

The aggregate over 58 fresh formal files is
`47197feb63d0fd4e65f6f97715c1456eb873937e3bc35fdcf1d0bc609a91a483`;
the aggregate over 147 diagnostic files is
`f5d4136f9bac400af456da58691f881ee902d88f065a22d0a933170b2dfbef1b`.

## Interpretation limits

Wall-clock measurements are retained as platform observations, not numerical
invariants; FEA counts are the reproducible cost measure. Restart and multistart
results show deterministic local-path variation and do not prove global
optimality. The fixed-layout mesh table repeats only response analysis: the rib
layout, thicknesses, and volume are held fixed.

## Historical exclusions

The following locally retained files predate the v1.2.0 evidence run and are not
current publication evidence: `case4_refresh.json`,
`case4_diagnostics_refresh.json`, `lifting_summary.csv`,
`figure_provenance.json`, and the auxiliary local/Q8/mortar/solid convergence
outputs or scripts not named by `SHA256SUMS`. They are intentionally excluded
from `manifest.json` and `SHA256SUMS`; their presence in a working tree must not
be interpreted as inclusion in the v1.2.0 release evidence.

Likewise, the pre-existing `results/formal_run_20260805-130032.log` contains
workstation paths and must not be placed in a release archive.

Verify the compact set from this directory with:

```powershell
Get-Content SHA256SUMS | ForEach-Object {
    $expected, $name = $_ -split '  ', 2
    if ((Get-FileHash $name -Algorithm SHA256).Hash.ToLower() -ne $expected) {
        throw "SHA-256 mismatch: $name"
    }
}
```
