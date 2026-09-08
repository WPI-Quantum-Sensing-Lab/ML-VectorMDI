# - Publication Release

Minimal code and artifacts needed to replicate the published experiment: a
6-part (1 main model + 5-model ensemble) deep ensemble of curriculum-trained `AttentionDipoleModel` networks plus isotonic magnitude-calibration
post-processing and the synthetic-data pipeline used to build the training and
testing datasets.

## What you can do

Every script in `src/` has a config file in `configs/`.
Run everything from inside `src/`; pass `--config PATH` to use a different
config file, and any extra CLI flags override the config values.

| # | Task | Command (from `src/`) | Config |
|---|---|---|---|
| 1 | Generate synthetic source maps | `python generate_data.py` | `configs/generate_data.ini` |
| 2 | Blend with experimental noise at set SNRs | `python blend_noise.py` | `configs/blend_noise.ini` |
| 3 | Train a single model | `python train_model.py` | `configs/train_model.ini` |
| 3.5 | Fit the isotonic calibration | `python calibrate.py` | `configs/calibrate.ini` |
| 4 | Train a deep ensemble | `python train_ensemble.py` | `configs/train_ensemble.ini` |
| 5 | Simple single prediction | `python predict.py` | `configs/predict.ini` |

`python run_full_experiment.py` reproduces the published experiment end to end
(train the 6-member ensemble with crash-safe resume, then collect results);
use `--dry-run` first to smoke-test the pipeline.

## Layout

```
├── configs/                    One config file per task (see table above)
├── requirements.txt            Python dependencies (Python 3.11, CUDA 11.8 torch build)
├── src/                        All code (run scripts from inside src/)
│   ├── generate_data.py … predict.py      Task entry scripts
│   ├── run_full_experiment.py             Published-experiment driver
│   ├── model/                             Library code
│   │   ├── architecture.py                AttentionDipoleModel, losses, metrics
│   │   ├── training/                      Curriculum + deep-ensemble training
│   │   └── calibration/                   Inference + isotonic calibration
│   └── data_creation/                     Synthetic-data pipeline
│       ├── generation/                    Source-map generation
│       ├── snr/                           SNR blending with blank noise
│       └── utils/                         HDF5 storage + dataset helpers
├── data/blanks/blanks.h5       All 14,568 experimental blank noise sections
│                               (Bx, By, Bz — each 14568 × 100 × 100 float64)
└── models/
    ├── ensemble_manifest.json  Manifest listing the 6 member checkpoints
    ├── member_00 … member_05/  Final trained weights (seed 42 … 47)
    └── calibration/            Fitted isotonic calibration tables
        ├── calibration_table_isotonic.json          per-dipole-count regressors
        ├── calibration_table_isotonic_global.json   single global regressor
        └── calibration_summary.json
```

## Code map

| Piece | Files |
|---|---|
| Model architecture | `model/architecture.py` holds `AttentionDipoleModel` (CNN backbone + attention pooling + DipoleNet heads, ≈3.2 M params), `DipoleNetLoss`, direction/magnitude metrics, HDF5 dataset wrappers |
| Training | `model/training/ensemble.py` (deep ensemble over seeds; also provides `EnsemblePredictor` for inference) → `model/training/curriculum.py` (2-D baby-steps curriculum: dipole count × SNR, competence-gated advancement) with `model/training/utils.py`, `model/training/stage_runner.py` (data loading, `train_one_stage`), `model/training/artifacts.py`, `model/training/metadata.py` |
| Isotonic calibration | `model/calibration/survey.py` runs the model over test datasets, pools (true, predicted) magnitudes, fits R²-weighted per-dipole and global `sklearn.isotonic.IsotonicRegression` maps and writes the JSON tables. `model/calibration/predict.py` provides model loading/inference. Apply a table by interpolating `X_thresholds → y_thresholds` (`np.interp`) on predicted magnitudes |
| Source-map generation | `data_creation/generation/magnetic_moment_gen_mp.py` (synthetic multi-dipole field maps, multiprocess); `data_creation/generation/multi_source_maker.py` shows the publication's per-dipole-count generation sweep; `data_creation/utils/hdf5_storage.py`, `data_creation/utils/dataset_conversion.py` |
| SNR-scaled data maps | `data_creation/snr/snr_blend_and_sort.py` pairs each noiseless source map with a blank noise section from `blanks.h5` at the desired SNR levels and writes the blended datasets |

## Replicating the experiment

1. **Generate source maps**: edit `configs/generate_data.ini`, then
   `python generate_data.py`. The publication's per-dipole-count sweep
   (with distance constraints) is scripted in
   `data_creation/generation/multi_source_maker.py`.
2. **Blend to SNR-scaled datasets**: point `configs/blend_noise.ini` at a
   source dataset and at `data/blanks/blanks.h5`, then
   `python blend_noise.py`. Training expects folders named
   `training_<N>dipoles_Distance_snr_<S>` under `data/datasets/`
   (dipole counts 1–10 × SNR levels 0.05–20).
3. **Train**: `python run_full_experiment.py` (use `--dry-run` first) for the
   published (1+5)-member ensemble, or `python train_ensemble.py` /
   `python train_model.py` for config-driven runs. Published settings (default in the configs): 6 models (seeds 42–47), DipoleNet, mixup rate 0.1,
   α = 0.6 curriculum, 20 000 samples/stage,
   early-stopped ≤300 epochs/stage, batch 64, AdamW lr 3e-4.
5. **Fit the isotonic calibration**: point `configs/calibrate.ini` at the run
   manifest of the evaluated test datasets, then `python calibrate.py`; the
   fitted tables are written next to the ones shipped in `models/calibration/`.
6. **Predict**: point `configs/predict.ini` at a `dataset.h5` and
   `python predict.py` prints the predicted moment, ensemble uncertainty,
   and calibrated magnitude for one sample.

## Using the shipped models from Python

```python
import json, torch
from pathlib import Path
import sys; sys.path.insert(0, "src")
from model.training.ensemble import EnsemblePredictor

manifest = json.loads(Path("models/ensemble_manifest.json").read_text())
predictor = EnsemblePredictor.from_checkpoints(
    [Path(p) for p in manifest["member_final_checkpoints"]],
    device=torch.device("cpu"),
)
mean_moment, std_moment = predictor(x)   # x: [B, 3, 100, 100]
```

## Notes on scope

- Legacy/experimental architectures and the figure/evaluation
  tooling from the research repository are intentionally omitted.
- `PATH_TO/...` placeholders in the `configs/` files must be pointed at your
  local copies of the data before running.
- `blanks.h5` contains the noise maps; the shipped models were trained on datasets built from
  exactly these.
