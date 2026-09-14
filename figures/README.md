# Academic figures

- [Combined Burgers and KS](burgers_ks_comparison.pdf) — both PDEs, ground truth, FNO prediction, AFNO prediction, and both absolute errors.
- [Burgers](burgers_comparison.pdf) and [KS](ks_comparison.pdf) — model rows with Ground truth / Prediction / Absolute error columns.
- [Error curves](burgers_ks_error_curves.pdf) — first-five and full-100 MSE across all 100 validation trajectories.
- [Supplement](supplement_trajectories_000_007.pdf) — eight fixed-index examples, including less favorable cases.

PNG versions accompany the main PDFs. PNGs use 400 dpi; PDFs preserve vector text and axes. See [CAPTION.txt](CAPTION.txt) for a reusable scientific caption and [metadata.json](metadata.json) for the full scales, population statistics and original figure checksums.

Run `python -m afno figures --output outputs/figures` to regenerate the main four figures. The eight-page supplement is a ready-made artifact from the original exports; only trajectory-zero fields are bundled, so regeneration of that supplement is outside the small demo.

The AFNO label refers to the 500 + 60 + 40 epoch model. FNO uses 500 epochs. Results are single-seed validation comparisons and use shared, unclipped color scales.
