# v1.5.0 numerical-verification metadata

This compact package indexes all manuscript-referenced numerical evidence regenerated for `v1.5.0`. The frozen executable-source aggregate SHA-256 is `27740f239031896d2e8acae097e7af516bf01b4f1322df0c6d5f819e4dd35607` over 41 source files. Formal results, cross-case local one-factor checks, 13 restart/multistart groups, sensitivity checks, the fixed-layout mesh response, and complete-pool Cases I--II all identify this same source snapshot.

The formal four-case run is authoritative for reported case results. The manuscript definition of **Total** is cumulative analyses through the first rationalization pass; it equals 39, 94, 133, and 124 for Cases I--IV and excludes `further_rationalized`.

The one-factor assessment uses Cases III and IV independently at relaxation values 2.5%, 5%, and 10%. It is local and case specific. Restart/multistart and fixed-layout mesh response are separate verification families.

The existing full source test suite was reused exactly once for this frozen snapshot: 151 tests passed in 86.003 s with exit code 0. It was not rerun while assembling this metadata package.

## Compact files

| File | Contents |
| --- | --- |
| `formal_stage_summary.csv` | All 18 formal saved stages and both cumulative-total meanings |
| `local_one_factor_summary.csv` | Six Case-III/IV local relaxation diagnostics |
| `restart_group_summary.json` | All 13 groups and all 45 individual runs |
| `sensitivity_by_step.json` | Cases II--IV component summaries by variable space and step |
| `fixed_layout_mesh_response.csv` | One fixed five-rib layout on four response meshes |
| `full_pool_stage_summary.csv` | Six complete-pool stages for Cases I--II |
| `manifest.json` | Release, source, formal-run, test, inventory, provenance, and hash metadata |
| `SHA256SUMS` | SHA-256 digests for every compact file except itself |

The 233 raw files under `Code/diagnostics_v1_5_0/` have inventory aggregate SHA-256 `eb09e2fb97c9d6a8c968db52e0b583ad3e8bd5f1ba6befa4885bc186527ea179`. The 58 formal files under `Code/results/example_1/`--`example_4/` have inventory aggregate SHA-256 `45a15b312abbbe4766a45a0fc7f211e1681f2fcecc56130df76f9bb4ebc2611c`. The algorithm is recorded in `manifest.json`.

All JSON and CSV paths are slash-separated and project relative.
