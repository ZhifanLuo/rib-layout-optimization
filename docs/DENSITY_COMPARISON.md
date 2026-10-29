> Historical parallel-layer model. The current manuscript uses the extruded-solid
> method described in [EXTRUDED_DENSITY_COMPARISON.md](EXTRUDED_DENSITY_COMPARISON.md).

# Fixed-base / parallel-layer density comparison

This independent Python SIMP implementation is available for all five examples.
It does not modify the rib optimizer or overwrite any saved rib-layout result.
It compares a horizontal bonded material layer with vertical explicit ribs;
this difference prevents interpreting compliance differences as an optimizer ranking.

## Model and matched inputs

The original Q4 MITC4/Reissner鈥揗indlin base shell, material, formal grid, loads,
supports, normalized load-case weights and mirror symmetries come from each
`exampleN.py` through `load_case`. All loads remain on the base reference plane;
pressure is integrated consistently, including supported-node contributions.
The base thickness and stiffness are fixed throughout optimization.

The design shell is horizontal, immediately above the base with no gap. Its
default solid thickness is `rib_height - base_thickness/2`, so its upper surface
has the same z coordinate as the original rib tops. An explicit positive
`--layer-thickness` can override this choice and is recorded. The layer occupies
`[base_thickness/2, base_thickness/2 + layer_thickness]`; its midsurface is at
`z=(base_thickness+layer_thickness)/2` relative to the base midsurface.

Perfect bonding and common shell rotations give the nodal relation
`u_layer = u_base + theta cross (0,0,z)`. In particular,
`ux_layer=ux_base+z*theta_y` and `uy_layer=uy_base-z*theta_x`.
The stiffness is the fixed base stiffness plus assembled
`[Emin/E0 + (1-Emin/E0)*rho_phys^p] * T.T @ K_layer_solid @ T`.
This is a bonded layer model, not a floating cover sheet or independent top skin.
The equal-rotation assumption and local thickness-to-feature ratios limit its
physical fidelity; a 3-D solid model may be needed near abrupt thickness edges.

## Design and optimization

One raw density in [0,1] belongs to each design-layer cell. Mirror averaging is
followed by a row-normalized linear physical-density filter. The default radius
is 1.5 times the larger cell dimension, in mm; `--filter-radius` is an explicit
physical length. Density refinement studies should retain the same physical
radius if feature-size effects are to be separated from mesh effects.
SIMP defaults to p=3 and Emin/E0=0. The fixed supported base already supplies all structural DOFs and stability, so no artificial design-layer stiffness is needed at void. A nonzero `--emin-ratio` is optional and recorded; it can noticeably stiffen very thin bases and should be assessed before comparison. The fixed base is never penalized.
Sparse-filter outputs within 64 floating-point epsilons of [0,1] are clipped to
the exact bounds to remove normalization/summation roundoff at fully solid
cells. Larger violations are rejected. Raw inputs and directly supplied physical
densities retain strict [0,1] validation; this repair does not relax their bounds.
The volume is `sum(rho_phys * cell_area * layer_thickness)` and its upper bound
is the original additional rib-volume budget; base volume is excluded from it.
An oversized budget is rejected rather than silently clipped to layer capacity.

The weighted compliance objective uses one factorization for all load cases;
Example IV weights are normalized to 0.5 and 0.5. Objective and physical-volume
sensitivities use the exact filter/symmetry adjoint. OC updates use move bounds
and bisection on actual filtered volume. Three consecutive iterations must meet
all three stopping tolerances: maximum raw design-variable change <=.01,
maximum physical-density change <=.005, and relative-objective change <=1e-4.
The raw-variable measure matches the one used by the public `top88` code
(Andreassen et al., 2011, section 3.3 and code line 81). The physical-density and
objective checks and the three-consecutive-step requirement are additional
conditions of this implementation. The default move limit is .1 instead of
the reference code's .2; this is an explicit algorithm setting, not a different
OC formula. Multiplier bisection here uses 100 steps to enforce the actual
physical volume, whereas `top88` uses a relative multiplier-bracket tolerance.
Reaching the iteration cap is reported separately.
No global optimality is claimed, and SIMP local minima depend on initialization,
penalization, filter radius and mesh. Initial density is uniform at the budget.
Entirely void filter neighborhoods have zero SIMP sensitivity for p>1 and cannot
re-enter under OC alone; the small raw-density update seed is not a global
escape mechanism from such local minima.

Earlier runs checked only physical-density and objective changes. Their stored
`converged` flags describe those former criteria and do not establish the new
raw-variable criterion. Filtering can suppress physical changes while raw
variables still move. The reference audit found that the saved earlier fine-grid
final states admitted a further OC step with raw changes above .01; they must
not be relabeled as executions of this corrected stopping rule. Fresh results
must record `raw_change` in every history row and the explicit tolerance in their
input specification.

The usual structural solve is kept unchanged when its original-equation relative
residual is at most 1e-7. Only a numerical failure or a larger/nonfinite residual
triggers a density-module-local fallback: diagonal equilibration solves
`D K D y = D f`, with `D=diag(K)^(-1/2)`, and recovers `u=D y`. Up to four
corrections use the original residual `f-Ku`. Acceptance always checks the
unmodified equation `K u=f` against the same 1e-7 threshold; neither diagonal
regularization nor a weaker tolerance is introduced. Final response records
identify whether this fallback was used and how many corrections were required.

## Commands and output

From `Code`, use a fresh output directory outside the source tree:

```text
python -B tools/run_density_comparison.py --cases 1 2 3 4 5 --output ../density-runs/full
python -B tools/run_density_comparison.py --cases 1 2 3 4 5 --output ../density-runs/full --execute
```

The first command only prints a plan. The second runs cases sequentially with
their full original meshes, 300 OC iterations maximum per case. `--mesh NX NY`
is an explicitly marked diagnostic override. Other options include
`--filter-radius`, `--layer-thickness`, `--penal`, `--emin-ratio`,
`--max-iterations`, `--move`, `--density-tolerance`, `--raw-density-tolerance`,
`--objective-tolerance`,
`--linear-solver` and `--no-images`. No resume/overwrite is performed.

The low-level Python `optimize(initial_raw=...)` API can initialize a new run from
a supplied raw-density array. It strictly validates shape, finite values and
[0,1] bounds, copies and mirror-averages the input, checks its actual filtered
volume against the budget, and reanalyzes the starting state. Initialization type
and starting physical volume are returned explicitly. This API is intended for
traceable continuation studies; the caller must record seed-file hashes and
original run provenance. Its initial response is a supplied-state response,
not a uniform-design response. The CLI exposes no seed option and always starts
uniformly at the budget. Neither API nor CLI resumes or overwrites old files.

Each case saves initial and final responses, per-load compliances and maximum
downward base displacement, actual design volume and equilibrium residual,
SIMP/filter/model settings, raw and physical density, displacement, CSV history,
and continuous-density/history/displacement figures. History distinguishes raw
design-variable change from physical-density change. Whole-run source fingerprints, input
specifications and timestamps are in `manifest.json`; case checkpoints retain
failure details and the last completed iteration. A completed process can still
have an `iteration_limit` termination and must not be described as converged.

The CLI now reports only the continuous SIMP solution. It does not threshold or
binarize the density, run postprocessed fixed-design analyses, save binary arrays,
or add a binary comparison. Historical outputs are preserved with their actual
source identities. The standalone `binary_design` utility remains available for
compatibility but is not part of the current run or comparison protocol.

Zero-density elements have exactly zero design-layer stiffness by default; an
explicit nonzero Emin override is reported. Isolated material patches and grey
regions are possible; connectivity, draw direction, minimum thickness, strength,
buckling and manufacturing constraints are not imposed. No rib count is assigned
to a horizontal material map. Reanalyze saved rib layouts separately under their
own documented model, and identify the geometry/volume conventions when comparing.

The density method and any new result are not part of the historical v1.6.0
release until an independently reviewed later publication explicitly includes them.

## Method references and attribution

The implementation follows the established SIMP material-interpolation approach
of Bendsøe and Sigmund (1999), the compliance/volume OC structure illustrated by
Sigmund (2001), and the **density-filter** formulation described by Andreassen
et al. (2011). It uses the physical-density filter and its exact chain rule for
both compliance and volume, not the alternative sensitivity filter. These
publications supply methodological references; no third-party source code is
copied into this implementation.

- Bendsøe, M. P., Sigmund, O. (1999). Material interpolation schemes in topology
  optimization. *Archive of Applied Mechanics* 69(9–10), 635–654.
  https://doi.org/10.1007/s004190050248
- Sigmund, O. (2001). A 99 line topology optimization code written in Matlab.
  *Structural and Multidisciplinary Optimization* 21(2), 120–127.
  https://doi.org/10.1007/s001580050176
- Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., Sigmund, O. (2011).
  Efficient topology optimization in MATLAB using 88 lines of code.
  *Structural and Multidisciplinary Optimization* 43(1), 1–16.
  https://doi.org/10.1007/s00158-010-0594-7
- Bathe, K.-J., Dvorkin, E. N. (1985). A four-node plate bending element based
  on Mindlin/Reissner plate theory and a mixed interpolation. *International
  Journal for Numerical Methods in Engineering* 21(2), 367–383.
  https://doi.org/10.1002/nme.1620210213

The 2-D continuum element used in the compact reference examples is replaced
here by the existing six-DOF MITC4 shell element and the explicitly derived
rigid-offset bonded-layer map above. The references are not claimed to publish
this identical two-layer model. Analytic rigid-body, extension and combined
extension/bending tests check that substitution, in addition to finite-difference
checks of the density/filter sensitivities. Skew affine-element membrane patches,
in-plane shear/spin, and arbitrary spatial orientation also check the shell's
coordinate-gradient mapping. The reference audit corrected its inverse-transpose
Jacobian mapping for skew elements; the rectangular elements used in the five
saved density grids have unchanged stiffness matrices to roundoff. Bibliographic entries are supplied
in `references/density_topology.bib` for later manuscript integration; the
manuscript bibliography is unchanged by this implementation. Bathe and Dvorkin
provide the MITC4 bending/shear basis; that reference does not specify all the
membrane and drilling stabilization choices in the existing shell implementation.
