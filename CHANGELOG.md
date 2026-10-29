# Changes

## v1.7.0 - 2026-10-29

- Update Example V and its normalized input to a 1.0 mm base, retaining the
  80 x 40 mesh and 5 mm rib-top height from the base midsurface. Supply the
  accepted 158-evaluation run ending with six ribs and two centre-line crossings.
- Add the Python shell-base/extruded-solid density model, shared-column design
  variables, SIMP/filter/OC implementation, references and validation tests.
- Correct the Q4 physical-gradient transformation for affine skew elements
  and include the corresponding patch tests.
- Add conforming extruded-solid rib evaluation for matched-volume comparisons
  with density layouts from Examples III and V.
- Supply saved density fields, histories, displacement fields, exact prepared
  rib meshes, executed numerical adapters, and portable replay/plot commands.
- Include direct-junction verification and six/eight-rib equal-volume regular
  references for the revised V. Preserve original configurations and source
  fingerprints of all accepted runs.
- Complete the I-IV result bundle and index alongside the current V and density
  evidence. Keep earlier source tags and historical verification unchanged.

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
