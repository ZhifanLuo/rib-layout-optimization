# v1.4.0 numerical-verification evidence

This directory indexes the complete successful numerical-diagnostic archive regenerated from the fresh formal v1.4.0 stages. The frozen executable-source aggregate SHA-256 is `3b06e4dab1fedffb90af0c7b3ee2c07676fb02cd3a87d20d4d45de3a7d99f6a0`. The archive contains 144 raw numerical files (81 strict JSON and 63 CSV), three sensitivity studies, 15 Case-II five-factor runs, 13 restart/multistart groups containing 45 runs, complete-pool Cases I--II, and a four-level fixed-layout response sequence through 160x80.

The formal four-case outputs are authoritative. Sensitivity, robustness, restart, full-pool, and fixed-layout checks were all regenerated from the same frozen executable source and fresh formal stages where applicable. The 15 raw robustness executions contain five equivalent reference-setting executions; their identical stage arrays are collapsed to one reference row only for manuscript display, giving 11 displayed outcomes. Raw records remain separate.

The initial robustness attempt affected only stdout delivery and is excluded from the numerical archive. Its disposition is recorded in `manifest.json:failed_attempts`. The successful retry has 15 complete records, zero failed records, empty stderr, and a complete final aggregate; its detached Windows process exit code was not recoverable and is not fabricated.

## Compact files

| File | Contents |
| --- | --- |
| `formal_stage_summary.csv` | Every stored stage in the four formal results |
| `sensitivity_by_step.json` | Cases II--IV finite-difference summaries by variable space and step |
| `robustness_summary.csv` | All 15 raw runs plus the explicit 11-outcome display mapping |
| `fixed_layout_mesh_response.csv` | One fixed five-rib layout on 20x10, 40x20, 80x40 and 160x80 meshes |
| `restart_group_summary.json` | All 13 restart/multistart groups and 45-run distributions |
| `full_pool_stage_summary.csv` | Complete 36- and 105-candidate pool stages for Cases I--II |
| `manifest.json` | Version, commands, provenance, raw inventory, hashes and failed-attempt disposition |
| `SHA256SUMS` | SHA-256 digests for the compact evidence set |

Formal `results.json` SHA-256 values:

- Case 1: `454f911766e3b3983a599a02e259de0bdab68eebd061464986b849a60b1905cc`
- Case 2: `aab7c044d507853020bdb43e57e529604262fb9c9b65ef72d4a91b50a8002299`
- Case 3: `a2fbe7d03f1bc6cc132dde6eaa3e8289dc7df7ae4f14fbfb97988d92c244a7d0`
- Case 4: `d0323d72ee3d1810770d64228cdd792a4ea6533ec6795deb4f9e876fa5cac033`

Raw diagnostic aggregate SHA-256: `12a40d5386eb757ebf337a5f89db85a63e56230deb9ac27145b2a9454154bc11`.

Verify this compact directory from PowerShell:

```powershell
Get-Content SHA256SUMS | ForEach-Object {
    $expected, $name = $_ -split '  ', 2
    if ((Get-FileHash $name -Algorithm SHA256).Hash.ToLower() -ne $expected) {
        throw "SHA-256 mismatch: $name"
    }
}
```
