# Differentiation And Contribution

## Differentiation

- End-to-end workflow: dataset layout, training, inference, evaluation, GUI, and
  Hugging Face model distribution are in one small repo.
- Non-developer client: meteorology users can run a local GUI without touching
  Python commands after setup.
- General-to-local strategy: the project is designed around a public base model
  plus regional fine-tuning and calibration.
- Benchmark-first framing: outputs can be evaluated with nowcasting metrics such
  as CSI, POD, FAR, HSS, ETS, RMSE, and FSS.
- Lightweight baseline: the first model is intentionally simple enough to train
  on a single RTX 3090, making it accessible for students and independent
  researchers.

## Open-Source Contribution

This project can contribute to the community by making the practical glue code
around rain-map nowcasting easier to reuse:

- A clean starter implementation for grayscale rain-map sequence prediction.
- A reproducible demo model that verifies the whole training-to-client pipeline.
- A benchmark CLI that lets users compare base and fine-tuned models.
- Documentation that connects local datasets to public benchmark practice.
- A path for regional users to adapt the model instead of starting from scratch.

## Honest Limitations

- The current public/demo weights are synthetic-data only.
- The current model is deterministic and does not produce uncertainty estimates.
- Pixel values are normalized grayscale intensities unless the user calibrates
  them to physical rainfall units.
- Real-world results must be reported per dataset, region, threshold, and lead
  time.

## Near-Term Research Milestones

1. Add real `.npz` radar-rainfall adapters, starting with Meteo-France
   `fr-radar-rainfall`.
2. Add checkpoint warm-start with `--resume-from` for true fine-tuning.
3. Add lead-time-specific metric tables.
4. Add baseline comparisons against persistence and optical-flow extrapolation.
5. Add optional auxiliary channels: latitude/longitude, elevation, land/sea
   mask, radar coverage mask, and quality flags.
