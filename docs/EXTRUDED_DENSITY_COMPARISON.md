# Fixed base shell and equal-height extruded solid density optimization

`rib_layout_algorithms/density_solid.py` adds an independent comparison model.
It is included in v1.7.0 alongside the rib optimizer. Historical results and
published v1.6.0 data retain their original identities. The parallel-shell model
is retained for its shared filter/OC implementation and earlier diagnostics;
the current paper compares the extruded-solid layouts described here.

## Geometry and independent unknowns

The base is the unchanged six-DOF Q4 MITC4 shell used by the examples. Its
material, thickness, mesh, base-reference-plane forces/pressure, normalized
load-case weights, supports and prescribed density mirror symmetries are
loaded from the actual example configuration.

The design domain consists of trilinear Hex8 solid elements with full2x2x2
Gauss integration. Its lower surface is at `z=base_thickness/2`; its upper
surface is at `z=config['rib']['height']`. The latter is the actual top
coordinate in the existing rib-shell model, measured from the base midsurface.
Thus the extrusion depth is `rib_height-base_thickness/2`. The model does not
reuse the different height convention of the older response-only
`StructuredQ1SolidReferenceModel`; only its rectangular Hex8 element kernel is
reused.

At the lower solid face, the translations are constrained to the physical top
surface of the base shell:

```text
ux_solid = ux_base + (base_thickness/2)*theta_y_base
uy_solid = uy_base - (base_thickness/2)*theta_x_base
uz_solid = uz_base
```

The bottom-brick constraint is applied by exact congruence `T.T @ Ke @ T`.
Only these interface translations are tied. Every higher solid plane has its
own three displacement unknowns per node. Neither rotations nor translations
are forced to be common along the height. Hybrid unknown ordering is the
original base shell DOFs, then consecutive solid planes from bottom to top.
The interface plane itself is eliminated by the constraint.

Original support constraints act on the original base shell DOFs only. The
solid side faces and top remain free. In particular, an originally clamped
base edge does not silently become a clamped vertical solid face. All loads
remain on the original base reference plane. The class exposes `ndof`,
`base_ndof`, `free_dofs`, `fixed_dofs` and padded `load_vectors` for external
equilibrium checks.

## Design and objective

One raw variable belongs to each planform cell. All Hex8 elements above that
cell share its physical density. The map is
`rho_ijk = (F P x)_ij`, where P is mirror averaging and F is the row-normalized
two-dimensional distance filter. This is a **three-dimensional mechanical
analysis with a two-dimensional column design space**, not an unrestricted
three-dimensional density field.

The constant-through-height condition gives equal-height straight extrusion,
consistent with the common rib height. It excludes overhangs along the chosen
z direction in the idealized extruded geometry. It does not impose a positive
draft angle, fillets, minimum wall width, connectivity or a complete casting
process model.

The fixed base stiffness is unpenalized. Added solid stiffness uses
`E/E0 = emin_ratio + (1-emin_ratio)*rho_phys**penal`, with default p=3 and
Emin/E0=1e-9. Unlike the old parallel-shell baseline, the upper solid nodes
remain in the system at void. A strictly positive, explicitly recorded floor
is therefore required; zero is rejected. The floor is not counted as physical
material volume. Its response effect must be assessed for very thin bases,
for example by reanalyzing fixed fields with1e-10 and1e-8. The default is a
documented numerical model parameter, not a certified negligible effect.

The weighted compliance is `sum_m w_m f_m.T u_m`, with weights normalized as
in the examples. Each column's sensitivity sums all of its z-element energies
before applying `-p*(1-emin_ratio)*rho**(p-1)`. No extra height-element count is
inserted. The actual physical volume is
`sum_ij rho_phys_ij*dx*dy*extrusion_depth`.

Filter, exact symmetry/filter adjoints, actual-volume OC bisection, move0.1 and
stopping logic are inherited unchanged from the separately audited
`ParallelLayerModel`. Three successive iterations must satisfy raw-variable
change<=0.01, physical-density change<=0.005 and relative-objective change<=1e-4.
Iteration-cap termination is distinct from convergence. The default CLI starts
uniformly at the configured physical budget.

The usual solver and density-local diagonal-equilibration/refinement fallback
are reused. Acceptance always tests the original reduced equation with a
relative residual<=1e-7. No additional diagonal penalty, weaker residual
tolerance, or matrix regularization is inserted.

Both new models accept `solver_threads=1` by default. An explicit larger
positive integer controls only MKL/Pardiso within the calling thread, using
`MKL_Set_Num_Threads_Local` from the library already loaded by pypardiso.
The original solve, equilibration and refinement all execute inside that
scope; the previous thread-local setting is restored in `finally`, including
on exceptions. Neither process-wide thread settings nor existing shell-model
source is modified. Counts>1 are rejected when Pardiso or its thread-local
MKL control is unavailable; SuperLU is not parallelized by this knob. The
existing automatic SuperLU fallback remains and is identified as such in the
control metadata, so a requested MKL count is not a claim that every solve
used that backend or attained parallel speedup. Requested thread count and
control scope are recorded in model metadata and response solve details.

## Fixed rib footprint replay and fair comparison

`evaluate_occupancy(area_fraction)` and `occupancy_stiffness(area_fraction)`
accept a fixed physical rib footprint on the same xy grid. They bypass density
filtering and the SIMP power. Fractional geometric cells use the explicitly
linear solid/void mixture
`emin_ratio+(1-emin_ratio)*area_fraction` throughout the extrusion height.
This is a geometric/Voigt approximation, not an exact conforming solid rib or
exact cut-cell quadrature. Subcell narrow ribs require planform refinement and
geometry-resolution checks before quantitative conclusions.

`rib_footprint_occupancy(config, ribs, samples=12)` is an optional diagnostic
midpoint-sampling helper for the union of finite flat-ended rectangular ribs.
Crossing material is counted once. It is not an exact geometric clipping
routine; callers can instead supply exact clipped-cell union areas. Boundary
clipping and intersections must follow the physical union, not summed strip
areas or capsule-shaped endpoints.

Original nominal rib volume is based on the rib optimizer's own conventions.
It can differ from the physical union volume above the base, both because
intersection overlap is counted differently and because extrusion starts at
the base's top surface. The replay always reports its actual physical volume
and never silently rescales geometry. A controlled common-model comparison
must record the nominal original budget and the represented physical volume
separately, and use the same actual additional-material budget for density
optimization. A caller can set `config['volume_bound']` explicitly to that
matched physical budget before constructing the model; the complete config
must be recorded in the experiment manifest.

`evaluate_physical` returns the full hybrid displacement and an explicit
`base_displacement` array, original-equation residual, per-load and weighted
compliances, physical column gradient and actual volume. `evaluate_occupancy`
returns the same response structure and identifies its separate linear
material representation. Its response is not an optimized/binarized density
design and should never be described as such.

For a resolved physical-rib comparison, the response-only
`solid_extruded_reference.ConformingExtrudedSolidModel(config, nodes_xy,
triangles, vertical_elements=8)` accepts an externally prepared conforming
triangular mesh of the exact clipped material union. It extrudes that mesh
into full-material6-node wedges, with3-point triangle times2-point height
quadrature. It contains no void-domain DOFs and therefore needs no ersatz
stiffness floor, filter or SIMP mixture. The supplied nonoverlapping CCW
triangles must cover the intended welded footprint; the model validates
indices, orientation, finite coordinates, duplicate triangles and used nodes,
but geometric coverage/nonoverlap remains the external mesher's contract.
No meshing dependency is imported by the solver.

Only lower-face translations are constrained by nodewise Q4 interpolation
and the shell-top offset. Higher nodes remain independent. This is an exact
constraint congruence at the supplied interface nodes; the triangular and
Q4 displacement traces between nodes differ. Interface refinement is needed
to assess this nonmatching-trace approximation. `analyze()` returns full/base
displacement, actual volume, per-load compliance and the original-equation
residual. Public sparse `stiffness`, `added_stiffness`, `constraint`, DOF and
load arrays permit independent saved-displacement reconstruction. Wedge
bending/height and footprint/interface refinement remain necessary even
though subcell material smearing has been removed.

## CLI

From `Code`, use a fresh output directory outside `Code`:

```text
python -B tools/run_extruded_density_comparison.py --cases 1 2 3 4 5 --output ../solid-density-runs/full
python -B tools/run_extruded_density_comparison.py --cases 1 2 3 4 5 --output ../solid-density-runs/full --vertical-elements 8 --max-iterations 2000 --execute
```

The first command plans without constructing a model, solving or creating the
output directory. The second executes sequentially in isolated new case
folders. The CLI uses each example's original grid and configured volume
budget unless an explicit diagnostic `--mesh NX NY` is supplied. Other
recorded controls are `--vertical-elements` (default8), `--filter-radius`
(physical mm), `--penal`, `--emin-ratio`, `--max-iterations`, `--move`, all three
stopping tolerances, `--linear-solver` and `--no-images`. Grid refinement should
retain the same physical filter radius when isolating mesh effects.
`--solver-threads N` records the optional MKL/Pardiso calling-thread count;
default1 preserves the original deterministic policy. Runtime availability
is checked when the model is built, not by read-only planning.

Each case saves raw/physical density, full hybrid and base displacement,
continuous density/history/base displacement figures, JSON responses, CSV
history and density, checkpoint and file inventory. The overall manifest
records exact inputs, source/entry hashes and process status. Existing output
directories are rejected. Continuous density is the sole optimized result;
there is no thresholding, binary response, binary file or binary comparison.

## Verification and scope

Focused tests check exact interface rigid rotation, assembled full-solid affine
3-D elastic energy, all six added-solid rigid modes, independent upper-plane
unknowns, all-variable compliance/volume finite differences with two load cases
and both symmetry axes, load/support/volume semantics, positive-floor void
equilibrium, linear unfiltered occupancy, rectangle-union semantics, inherited
OC histories and continuous-only CLI execution. Independent auditing and
controlled fixed-state checks must precede numerical comparison claims.

Full-integration trilinear elements can lock in slender bending problems and
must resolve both planform features and height. A4/8/16 height-layer study is
useful, but does not replace planform convergence or exact geometry resolution.
Likewise, a small equilibrium residual verifies the discrete solve, not
continuum accuracy. The base remains a shell approximation and its interface
tie remains an explicit modeling assumption. No global optimum, mesh
independence, identical layout, or superior algorithm is claimed in advance.

## References

The SIMP/filter/OC foundations remain Bendsøe and Sigmund(1999),
Sigmund(2001), and Andreassen et al.(2011), collected in
`references/density_topology.bib`. The MITC4 base-shell foundation remains
Bathe and Dvorkin(1985). Additional references are supplied in
`references/density_extruded_topology.bib`:

- Liu,K., Tovar,A.(2014). An efficient3D topology optimization code written in
  Matlab. *Structural and Multidisciplinary Optimization*50,1175–1196.
  https://doi.org/10.1007/s00158-014-1107-x . Its author-provided Top3d code
  illustrates H8/SIMP, positive void stiffness, density filtering and OC for
  three-dimensional analysis. Our shell coupling and exact column-variable
  map are additional modeling choices, not an implementation attributed to
  that paper.
- Wang,B., Zhou,Y., Tian,K., Wang,G.(2020). Novel implementation of extrusion
  constraint in topology optimization by Helmholtz-type anisotropic filter.
  *Structural and Multidisciplinary Optimization*62,2091–2100.
  https://doi.org/10.1007/s00158-020-02597-1 . This supports the extrusion
  constraint concept and discusses traditional design-variable mapping. The
  present implementation uses exact column mapping; it does not implement
  their Helmholtz anisotropic-filter algorithm.

These publications do not validate the exact shell–solid interface, footprint
mixture or support semantics implemented here. Those choices are separately
specified and tested. No external reference code was copied into this new
optimizer; the existing project Hex8 kernel and audited project optimization
methods are reused.
