# Changes

## v1.6.0 — 2026-10-26

- Add consistently integrated uniform Q4 pressure loading and the paper's revised
  Example V, now using the same standard CLI/output schema as I–IV and defaulting
  to the 0.5 mm / 80 × 40 manuscript model.
- Supply the complete archived V configuration and a dry-plan/full-run helper with
  fresh-output guards and source/configuration/result fingerprints.
- Include controlled insertion and simplification drivers, direct-deletion
  policies with a single thinnest-group fallback, parameter checks and regression
  tests for pressure and symmetry-aware selection.
- Permit the first complete four-rib mirror group under two-axis symmetry with the
  default two-seed setting; preserve global sizing acceptance and later-seed caps.
- Extend sensitivity validation to the pressure case and distinguish preserved
  traces from discrete trace changes.
- Document the versioned paper-data asset, current Figure 8 source, historical
  provenance, third-party licenses and external-method rerun setup.
- Preserve historical verification records without relabeling them as new runs.
  Allow the diagnostic metadata test's documented no-Git source-archive state.

## v1.5.0 — 2026-08-13

Historical rationalization-threshold release. Its source and associated
verification remain fixed under tag v1.5.0; consult that tag for original details.
