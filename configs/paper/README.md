# Manuscript input configurations

`example5.json` is the complete, normalized input for the v1.7.0
manuscript Example V, equal to `load_case(5)`. Its base thickness is 1.0 mm;
the base mesh is 80 by 40 on a 100 by 50 mm plate. The rib top remains 5 mm
above the base midsurface, or 4.5 mm above its upper face.

The v1.6.0 release used the earlier 0.5 mm base. Its unchanged historical input
and result are supplied in the revision-data release asset at:

`agent_work/runs/20261019-103343-case5-fine-mesh/experiments/002_t0.5_80x40/example_5/results.json`

That saved result is supplied in the v1.6.0 revision-data release asset. Its
SHA-256 is
`122457131d53335d82d950930f74c488f29dd069f2f762185e1f78d471b19afc`.

The released source and results are not relabeled as the new 1.0 mm calculation.
The earlier 1 mm / 40 by 20 case is also historical.

Run `python -B tools/reproduce_paper.py --output ../paper-V` to inspect the plan;
append `--execute` to run it in a fresh directory outside the source tree.
The helper records this JSON file's hash separately from the executable-source
hash and checks both remain unchanged during execution. It uses no quick-mode or
iteration-limit overrides. The configuration is input, not fitted output data.
