# Reproduce the v1.7.0 paper results

Download the revision-data ZIP from the v1.7.0 GitHub release and extract it
into a new directory. Check `SHA256SUMS.txt` before extraction and the inner
`SHA256SUMS.json` for individual files. The data root contains:

- `Code/`: this release's source snapshot.
- `Code/results/`, `Code/diagnostics_v1_5_0/`, and dated `agent_work/runs/`:
  accepted I-IV optimization and controlled-study evidence.
- `Supplementary_Density_Comparison.zip`: the immutable current
  V and III/V comparison package, with original execution snapshots.
- `release_verification/`: source inventories, test results and packaging checks.
- `README.md` and `EVIDENCE_INDEX.md`: current-result authority and commands.

From `Code/`, install `requirements.txt`, then run the five `exampleN.py`
entry points as shown in the source README. These are new optimizations; use
fresh output directories. `tools/reproduce_paper.py` checks the exact V input.

For the selected density/rib comparison, extract the nested comparison ZIP
into a new folder named `comparison/`. From its extracted root:

```console
python -B plot_final_comparison.py --root historical --updated-root current/density --output ../redrawn-comparison
python -B reproduce_updated_v.py --mode both --output ../fresh-v-comparison
```

The first command draws Figure 8 without solving. The second validates the
hashed inputs/source and prints a run plan. Add `--execute` to replay V, or
select `--mode ribs` or `--mode density`. Follow `historical/README.md` for
the unchanged Example III. Follow `updates/regular6-8/README.md` for the
current six/eight-rib regular references and Figure 7.

The supplied prepared footprint meshes need no meshing library. Regenerating
meshes additionally needs Shapely and Triangle; see their upstream licenses.
Plotting needs NumPy and Matplotlib. Large solid solves use Pardiso/MKL with
eight thread-local solver threads in the saved run adapters. Exact solver
acceptance, mesh/volume inputs and stopping histories are recorded with the
results. Coarse initialization is optional; the paper reports the final mesh.

The comparison package contains every exact execution snapshot required by
its inventory checks. Do not replace an embedded Code tree with another
version. Its older README notes, earlier V studies and three/five-rib
references are archived records, superseded by the current release index.
The paper's V uses a 1.0 mm base, six optimized ribs and regular references
with six/eight ribs. The older external-method comparison is not used.

Saved current results include:

| Quantity | Example III | Example V |
| --- | ---: | ---: |
| Native final rib count | 12 | 6 |
| Common-model added solid volume (mm3) | 9911.7943 | 1302.0797 |
| Common-model rib compliance (N mm) | 25.1438 | 0.0034096494 |
| Density compliance (N mm) | 31.2767 | 0.0046151551 |
| Final density planform mesh | 160 x 80 | 160 x 80 |
| Height elements | 8 | 8 |

Example V's native shell compliance is 0.0032882400064259115 N mm. Its
six/eight regular-reference compliances are 0.004575347771776578 and
0.004456873192100891 N mm at equal additive rib volume 1500 mm3.
Native shell responses and common shell-solid responses use different
elements, base meshes and rib connection definitions; compare the paired
common-model results when comparing the two optimization methods.
