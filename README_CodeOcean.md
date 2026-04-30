# C-SPIKES Code Ocean Capsule Outputs

This capsule is a reproducibility demonstration for the C-SPIKES manuscript. It does not rebuild
the publication figures end-to-end. Instead, it reruns the core spike-inference mechanics on a
small, reviewer-runnable subset, records quantitative trialwise correlations, and writes checksums
for packaged inputs and regenerated outputs.

## How to Run

From the Code Ocean capsule terminal:

```bash
cd /root/capsule/code
./run.sh setup quickcheck smoke inference biophys-ml
```

For a short environment/GPU check only:

```bash
cd /root/capsule/code
./run.sh setup quickcheck smoke
```

The smoke stage writes GPU/backend visibility, a one-epoch inference plan, and walltime fields under
`/root/capsule/results/smoke/`.

## Output Crosswalk

| Manuscript output | Capsule stage | Main capsule outputs | What is reproduced or checked |
| --- | --- | --- | --- |
| Figure 4B,D | `inference` | `results/inference_parity/trialwise_correlations_jG8f_repro.csv`, `results/inference_parity/correlation_parity.csv`, `results/inference_parity/plots/cell2_jG8f_trace_panel.png` | jGCaMP8f sample-epoch inference traces and trialwise correlation-to-ground-truth checks for BiophysSMC, BiophysML, ENS2, and CASCADE. |
| Supplementary Figure 12A,B | `inference` | `results/inference_parity/trialwise_correlations_jG8f_repro.csv`, `results/inference_parity/trialwise_correlations_reproducible.csv` | jGCaMP8f downsampled correlation mechanics for raw, 30 Hz, and 10 Hz outputs where those rows are generated. |
| Supplementary Figure 13C,D | `inference` | `results/inference_parity/trialwise_correlations_jG8f_repro.csv`, `results/inference_parity/plots/cell3_jG8f_parameter_trace_panel.png` | jGCaMP8f PGAS/BiophysSMC comparison using the default gold GCaMP parameters versus the Janelia-derived jGCaMP8f parameter set. |
| Supplementary Figure 14F,G | `inference` | `results/inference_parity/trialwise_correlations_jG8m_repro.csv`, `results/inference_parity/plots/cell4_jG8m_trace_panel.png` | jGCaMP8m sample-epoch inference traces and trialwise correlation-to-ground-truth checks. |
| Figure 4A | `biophys-ml` | `results/biophys_ml_parity/biophys_ml_reproducibility_report.json`, `results/biophys_ml_parity/checksum_summary.csv` | BiophysML construction mechanics: packaged PGAS parameter samples, GCaMP parameters, synthetic ground-truth generation, and checkpoint checksum checks. |
| Figure 4B,D | `biophys-ml` | `results/biophys_ml_parity/trialwise_correlations_trace_check.csv`, `results/biophys_ml_parity/plots/biophys_ml_retrained_trace_panel.png`, `results/biophys_ml_parity/checksum_summary.csv` | Reference-versus-retrained BiophysML prediction trace parity and prediction checksum checks. |

## CSV Figure Column

Generated trialwise and checksum CSVs include a `manuscript_figure` column. The value is a
manuscript figure or panel label. It identifies the
figure whose computational mechanics are being demonstrated by that row.

Key CSVs:

- `results/inference_parity/trialwise_correlations_jG8f_repro.csv`
- `results/inference_parity/trialwise_correlations_jG8m_repro.csv`
- `results/inference_parity/trialwise_correlations_reproducible.csv`
- `results/inference_parity/correlation_parity.csv`
- `results/biophys_ml_parity/checksum_summary.csv`
- `results/biophys_ml_parity/trialwise_correlations_trace_check.csv`

Reference paper-summary CSVs staged in `data/reference_outputs/paper_summaries/` are included as
comparison targets. They summarize the manuscript-scale runs that the capsule subset compares
against; MATLAB MLspike rows are retained there for reference but are not rerun by this Python
capsule.

## Important Scope Notes

- The capsule intentionally runs a small subset by default so reviewers can verify the pipeline on
  Code Ocean hardware.
- The companion figure-construction repository performs the full intermediate-to-final figure
  assembly. This capsule focuses on reproducible code execution, trialwise quantitative outputs, and
  checksums.
- jGCaMP8f PGAS/BiophysSMC defaults use the slice-condition gold parameters
  (`data/pgas_parameters/20230525_gold.dat`). The Janelia-derived parameter file is used only for the
  explicit Supplementary Figure 13 parameter comparison.
