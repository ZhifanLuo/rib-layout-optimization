# Manuscript input configurations

`example5.json` is the complete, normalized `revision_config` copied from the
saved manuscript Example V run:

`agent_work/runs/20261019-103343-case5-fine-mesh/experiments/002_t0.5_80x40/example_5/results.json`

That saved result is supplied in the v1.6.0 revision-data release asset. Its
SHA-256 is
`122457131d53335d82d950930f74c488f29dd069f2f762185e1f78d471b19afc`.

The configuration uses a 100 by 50 mm plate, 0.5 mm base thickness and an 80 by
40 shell mesh. In v1.6.0, `load_case(5)` normalizes to this same complete input.
The earlier 1 mm / 40 by 20 case is historical and is not the manuscript default.

Run `python -B tools/reproduce_paper.py --output ../paper-V` to inspect the plan;
append `--execute` to run it in a fresh directory outside the source tree.
The helper records this JSON file's hash separately from the executable-source
hash and checks both remain unchanged during execution. It uses no quick-mode or
iteration-limit overrides. The configuration is input, not fitted output data.
