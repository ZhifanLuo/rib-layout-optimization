# v1.7.0 verification index

The complete unit suite ran 235 tests: 234 passed and one optional private-backup
parity check was skipped. The new tests cover density filtering/gradients/OC,
extruded-solid equilibrium and interfaces, conforming prism elements, and the
Q4 affine-gradient patch correction.

`manifest.json` lists all 68 numerical source/config/test/tool files and proves
that the published source matches the accepted executed snapshot after newline
normalization. It preserves both executed and canonical release hashes.

The separate v1.7.0 revision-data asset contains the accepted result files,
historical manifests, numerical audits, figure sources and portable commands.
Start with its root README and EVIDENCE_INDEX.md. Its current comparison package
retains original execution adapters and source hashes; do not replace embedded
snapshots with this repository tree when replaying those exact calculations.

No optimization was rerun merely for release packaging. Earlier execution dates
and source identities remain unchanged. Example V uses a 1.0 mm base and six
final ribs; the common-model density comparison uses Examples III and V.
