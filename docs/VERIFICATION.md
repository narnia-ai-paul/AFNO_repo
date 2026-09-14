# Release verification — 2026-09-14

The standalone package was built and installed into a separate Python environment. Demos and figures ran from outside the project with no `PYTHONPATH` pointing to the original research workspace. Temporary paths containing spaces were exercised. No full-budget training or GPU experiment was launched for this packaging task.

| Check | Result |
| --- | --- |
| Unit/integration suite against the installed package | **15 tests passed** |
| Full small Burgers and KS workflows | Passed: generate → FNO/AFNO base → warmup → calibration → balanced → evaluate |
| Interrupted base, warmup and balanced training | Bitwise-equal final model states versus uninterrupted runs in the tested CPU environment |
| Complete workflow resume | Reuses verified checkpoints; modified completed artifacts are rejected |
| CLI and multiprocessing generation | KS smoke command passed with two generator workers |
| Standalone evaluation command | Native-grid validation passed |
| All four ported models versus original frozen model source | **Bitwise-identical** 100-step predictions at the same inference batch size |
| Burgers and KS reference solver regeneration | **Bitwise-identical** bundled trajectory-zero initial and future fields |
| Pretrained demo versus original saved exports | Maximum absolute differences below 6.1e-5; tests use rtol=2e-4, atol=2e-4 |
| Main four PNG figures regenerated from bundled saved fields | **Identical pixels and SHA-256 hashes** |
| Asset checksums | All 13 bundled configuration/data/weight files verified |

The 15 tests additionally check the batched Euler update, one-time initial encoding, analytic gradients reaching the encoder from step 100, activation-checkpoint gradient equivalence, horizon-aware checkpoint selection, readout interpolation, and avoidance of test-file generation/evaluation.

Machine-readable evidence is in [port_verification.json](port_verification.json), [figure_verification.json](figure_verification.json), and [demo_verification.json](demo_verification.json). The demo metrics describe one fixed validation trajectory per PDE, not the full-population table in the README.

These checks establish code and artifact portability in the tested Python 3.11 CPU environment. They do not establish multi-seed scientific superiority or certify retraining outcomes on other hardware. The full 500/60/40-epoch training budgets were inherited as recipes and were not repeated for this release.
