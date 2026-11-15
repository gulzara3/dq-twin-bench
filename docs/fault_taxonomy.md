# Fault taxonomy

Twelve faults are injected into the test windows at inference time. Models are
always trained and calibrated on clean data. Values are in normalised sensor
units (z-scores). Severity runs from 1 (incipient) to 5 (gross failure), and
each fault and severity is realised with ten injection seeds.

| Code in the CSV files | Name in the paper | Family | Data-quality dimension | Severity parameters, levels 1 to 5 |
|---|---|---|---|---|
| `gaussian_noise` | Gaussian noise | Value | Accuracy | standard deviation 0.05, 0.1, 0.25, 0.5, 1.0 |
| `nongaussian_noise` | Heavy-tailed noise | Value | Accuracy | Student-t with 3 degrees of freedom, scale 0.05, 0.1, 0.2, 0.35, 0.6 |
| `bias_offset` | Calibration bias | Value | Accuracy | constant offset of 0.1 to 2.0 on 1 to 5 channels |
| `drift` | Calibration drift | Value | Accuracy, currency | linear ramp reaching 0.1, 0.2, 0.5, 1.0, 2.0 at the window end |
| `missing_mcar` | MCAR missingness | Completeness | Completeness | uniform deletion of 5, 10, 20, 35, 50% of values, forward fill |
| `missing_mar` | MAR missingness | Completeness | Completeness | deletion conditioned on an observed channel, 5 to 70%, forward fill |
| `missing_mnar` | MNAR missingness | Completeness | Completeness | deletion of the largest values, 5 to 70%, forward fill |
| `sensor_dropout` | Sensor dropout | Availability | Completeness, accessibility | 1, 2, 3, 5, 7 channels set to zero |
| `timestamp_jitter` | Timestamp jitter | Timing | Timeliness, consistency | 1, 2, 4, 8, 12 swaps of adjacent samples |
| `channel_desync` | Channel desynchronisation | Timing | Timeliness, consistency | 1 to 7 channels shifted by 1 to 6 cycles, edge hold |
| `sampling_reduction` | Sampling reduction | Timing | Timeliness | decimate and hold by a factor of 1, 2, 3, 4, 6 |
| `concept_drift` | Regime shift | Context | Believability, currency | affine change with gain 1.05 to 2.0 and offset 0.05 to 1.0 |

Where a range is shown, the five level values are set in the benchmark code
(see `benchmark/README.md`).

## Notes

- **Missingness mechanisms.** MCAR deletes values at random. MAR deletes values
  depending on the observed level of a reference channel. MNAR deletes the
  largest readings, so the loss depends on the missing values themselves.
  Deleted readings are forward-filled before the model sees the window.
- **Rate grids.** The MCAR grid ends at 50% loss, and the MAR and MNAR grids
  end at 70%. Comparisons at equal severity are therefore not exactly matched
  in rate.
- **Sampling reduction, severity 1.** A factor of 1 leaves the stream unchanged.
- **Channel selection.** Calibration bias and sensor dropout pick the affected
  channels at random, which is why their results vary most across injection
  seeds.

## Ranking by damage

Corruption Robustness Index (RMSE under the fault divided by clean RMSE),
pooled over severities, subsets, architectures and seeds:

| Rank | Fault | CRI |
|---|---|---|
| 1 | Calibration drift | 2.71 |
| 2 | MNAR missingness | 2.37 |
| 3 | Regime shift | 1.82 |
| 4 | MAR missingness | 1.62 |
| 5 | Calibration bias | 1.56 |
| 6 | Sensor dropout | 1.46 |
| 7 | Heavy-tailed noise | 1.31 |
| 8 | Sampling reduction | 1.29 |
| 9 | Gaussian noise | 1.25 |
| 10 | MCAR missingness | 1.09 |
| 11 | Channel desynchronisation | 1.04 |
| 12 | Timestamp jitter | 1.00 |
