# Revised-paper experiment guide — v1.6.0

Use the installed Python environment from the repository root. The revision driver
and paper helper below print plans unless `--execute` is appended. They require a
new output directory outside the source tree. Keep failed/interrupted results and
launch another output root; do not silently overwrite evidence. Omit diagnostic
`--iteration-cap` overrides for paper calculations.

## Current Example V

`example5.py` uses the manuscript's 100 × 50 mm base, 0.5 mm thickness, 80 × 40 Q4
mesh, 5 mm rib height, initial rib thickness 0.5 mm, upper 3 mm, 25 mm lattice,
1500 mm³ rib-volume limit and y reflection. E=70000 N/mm², nu=0.33. Uniform
pressure −0.0001 N/mm² gives −0.5 N total. Only the base right edge is clamped;
rib-top nodes stay free. The former 1 mm / 40 × 20 case is historical.

The complete normalized `configs/paper/example5.json` equals `load_case(5)`.
The helper retains its exact configuration and checks source/config fingerprints:

```console
python -B tools/reproduce_paper.py --output ../rerun/paper-V
```

Append `--execute` for the complete workflow. Separate stage-response replays do
not contribute to optimization FEA counts. All five standard entry points share
the same regular CLI; unlike the guarded helper, this command runs immediately:

```console
python -B example5.py --output ../rerun/standard-V
```

## Controlled tests

The revision driver records complete configurations, source/input/output hashes
and started/complete/failed checkpoints. `--resume` verifies complete trials with
matching fingerprints; it is not recovery of interrupted optimization. The
`--max-trials` value is a hard count guard, not truncation. Use `--no-images` for
single-stage comparisons because the four-stage plot is inapplicable.

```console
# Insertion ablation; use --case 5 for the current bending model.
python -B tools/run_revision_experiments.py --mode no-addition --case 2 --output ../rerun/noaddition-II

# Three simplification policies from one geometry-stage layout.
python -B tools/run_revision_experiments.py --mode compare --case 2 --source path/to/example_2/results.json --output ../rerun/deletion-II --max-trials 3 --no-images

# One factor at a time: quantile multipliers, then maximum beta.
python -B tools/run_revision_experiments.py --mode sweep --case 2 --source path/to/example_2/results.json --alpha-multipliers 0.5 2 --betas 10 --output ../rerun/alpha-II --max-trials 2 --no-images
python -B tools/run_revision_experiments.py --mode sweep --case 2 --source path/to/example_2/results.json --alpha-multipliers 1 --betas 5 20 --output ../rerun/beta-II --max-trials 2 --no-images

# Prescribed equal-volume longitudinal reference, not an optimized benchmark.
python -B tools/run_revision_experiments.py --mode longitudinal --case 5 --longitudinal-count 3 --output ../rerun/longitudinal-three --no-images
```

Append `--execute` to run. Replace `path/to/...` with a saved result of the requested
case. Comparisons replay saved geometry first and use the same absolute limit,
`1.05 × saved geometry compliance`, for all three policies. A separate final FEA
must meet the exact limit. Replay and final checks are excluded from the published
simplification count.

The continuous policy minimizes the projected thickness-threshold count. Direct
policies skip that solve, delete qualifying ribs and use joint endpoint/thickness
reoptimization or fixed-endpoint resizing. If no complete mirror group qualifies,
each direct policy tries the thinnest complete group once, ranked by mean input
thickness. Response validation/restoration determines acceptance. These are the
tested policies, not all possible greedy simplification strategies.

## Sensitivities and saved evidence

The release data asset maps every table/figure to saved inputs and results. This
example checks the first rib of III in individual and mirror-reduced spaces at the
two publication step fractions; the diagnostic executes immediately:

```console
python -B tools/run_sensitivity_verification.py --case 3 --source path/to/example_3/results.json --output ../rerun/sensitivity-III --verification-space both --rib-indices 1 --thickness-steps .01 .001 --endpoint-steps .01 .001
```

Case II uses rib indices 1 through 8; consult the evidence index for all sample
choices. Trace-preserving errors and trace-changing differences are reported
separately; these checks do not prove differentiability across discrete trace
changes.

Historical manifests can contain original machine paths and earlier models. Those
are provenance records, not current portable commands. Engineering reproduction
and Figure 8 source are documented in the release data asset. No experiment is
claimed to have completed merely because a command is listed here.
