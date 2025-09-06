# Evaluation: ADE, CRPS, and PIT

- **ADE** (Average Displacement Error): mean L2 distance between forecast and ground truth trajectory over the forecast horizon.
- **CRPS**: continuous ranked probability score; we approximate via Monte Carlo samples per time-step and dimension.
- **PIT histograms**: Probability Integral Transform across time/coords; near-uniform indicates calibration.

Outputs:
- `outputs/metrics.json`: ADE, RMSE, CRPS summary.
- `outputs/pit_hist.png`: PIT histogram.
- `outputs/qualitative.png`: trajectory overlays.
