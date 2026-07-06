# Understanding and Mitigating Distribution Shifts in Volumetric Lung Nodule CAD Using a 3D Vision-Language Framework

This repository contains the MedStyle-3DG implementation for a 3D nodule false-positive reduction framework that evaluates visual generalization across acquisition domains. The codebase is organized around the article protocol: train on source domains, select checkpoints only on the in-distribution validation split, and report final generalization on the held-out OOD split.

The repository contains only the framework code needed for preprocessing, data loading, training, validation, metrics, and inference-time evaluation.

## Environment

Use Python `3.12` and install the pinned direct dependencies:

```bash
python3.12 -m pip install -r requirements.txt
```

## Method

The training protocol has three sequential phases.

1. Phase 1 trains a 3D ResNet-50 visual encoder with MixStyle and dense SWAD using a validation-loss valley rule.
2. Phase 2 aligns the visual encoder with a frozen BiomedVLP-CXR-BERT text encoder using image/text contrastive similarity.
3. Phase 3 applies DPStyler-inspired style removal with residual Style-SE, ArcFace domain supervision, domain-uncertainty loss, validation-gated greedy model soup over the best style checkpoints, and final three-branch inference ensemble evaluation.

Phase 3 evaluates three inference branches:

- canonical image/text branch
- Style-SE branch
- style-consensus branch sampled from the source-domain style bank

Checkpoint selection is validation-only. `TestOOD.csv` is evaluated after the best validation checkpoint or validation-selected soup has already been chosen. OOD reports do not sweep an oracle threshold on the OOD split.

## Expected Data

The protocol expects a split directory with:

- `Train.csv`
- `TrainPlusTestID.csv`
- `Valid.csv`
- `TestOOD.csv`

Required columns:

- `patch_name`: relative `.npy` patch path under `--patch-root`
- `label`: binary target, `0` or `1`
- domain/style column, for example `origin` or `manufacturer`

Optional columns such as `ct_id`, `x`, `y`, `z`, and `length` are carried into batch metadata when present.

## Preprocessing

Use preprocessing when starting from raw CT volumes and a CSV containing nodule coordinates. The script reads DICOM directories, NIfTI, MHA/MHD, NRRD, NPY, or NPZ volumes; resolves spacing; replaces invalid HU values, clips HU to the configured window before and after resampling, resamples to the target spacing with NVIDIA DALI LANCZOS3 by default, crops around the provided center, and writes `.npy` patches.

Install `nvidia-dali-cuda120` for the default DALI backend, or pass `--resample-backend torch_trilinear` for the fallback implementation.

```bash
python -m preprocessing.preprocess_patches \
  --input-csv /path/to/input.csv \
  --output-csv /path/to/output.csv \
  --patch-root /path/to/patches \
  --ct-root /path/to/raw_cts \
  --target-spacing 1,1,1 \
  --patch-shape 64,45,45 \
  --image-column image_path \
  --ct-id-column ct_id \
  --x-column x \
  --y-column y \
  --z-column z \
  --coordinate-units voxel
```

The preprocessing output patch is intentionally larger than the model input. The training loader performs the final crop-or-pad to `32x32x32`.

## Loader And Augmentation

Patches are loaded from `.npy` files and converted to `[C,D,H,W]`.

Training uses domain-aware batching by default: every source-domain batch is balanced as far as the batch size and available samples allow.

Training applies, in order:

- random affine with scale, rotation, and translation
- random elastic deformation (`sigma=20`, `points=3`)
- crop-or-pad to `32x32x32`
- additive Gaussian noise in Hounsfield units (`sigma` sampled from `10-50 HU`) before intensity rescaling
- percentile intensity rescaling to `[0,1]`
- random left-right flips
- random Gaussian blur (`sigma` sampled from `0.5-1.5`)
- multiplicative intensity scaling

Install `elasticdeform` for the elastic-deformation augmentation.

Validation and OOD evaluation are deterministic:

- crop-or-pad to `32x32x32`
- percentile intensity rescaling to `[0,1]`

## Training Protocol

Run the full article protocol:

```bash
python training/run_article_protocol.py \
  --split-dir /path/to/splits \
  --patch-root /path/to/patches \
  --output-root /path/to/runs \
  --text-model /path/to/BiomedVLP-CXR-BERT-specialized \
  --trust-remote-code \
  --domain-column origin \
  --label-column label \
  --patch-column patch_name \
  --model resnet50 \
  --seed 1337 \
  --device-id 0 \
  --num-workers 8 \
  --phase1-epochs 300 \
  --phase1-batch-size 128 \
  --phase1-lr 0.001 \
  --weight-decay 0 \
  --phase2-epochs 100 \
  --phase2-batch-size 128 \
  --phase2-lr 0.0001 \
  --phase3-epochs 100 \
  --phase3-batch-size 128 \
  --phase3-lr 0.008 \
  --phase3-soup-size 5 \
  --style-k 20 \
  --selection-metric best_f1
```

The reported results average five independent runs with seeds `1337, 2022, 42, 2026, 0`; run the command once per seed and combine the outputs with `evaluation/aggregate_protocol_seeds.py`. The hyperparameters above are the command-line defaults, so the same run can be launched without passing them explicitly.

Add `--deterministic` when an audit run needs stricter CUDA reproducibility. It can reduce throughput.

The script runs both regimens:

- `Train -> Valid -> TestOOD`
- `TrainPlusTestID -> Valid -> TestOOD`

`TrainPlusTestID.csv` is treated as an expanded training set. It is not used as an evaluation target by the protocol.

## Outputs

Each run writes:

- `protocol_results.json`: per-epoch history, validation metrics, OOD metrics, thresholds, ensemble branch metrics, and checkpoint metadata
- `phase1_best.pt`: validation-selected visual checkpoint, possibly SWAD-sourced
- `phase2_best.pt`: validation-selected aligned MedStyle checkpoint
- `phase3_best.pt`: validation-selected greedy soup checkpoint
- `phase3_soup_members/`: individual Phase 3 candidate checkpoints considered by the greedy soup, with one candidate collected after each Phase 3 epoch validation

Greedy soup starts from the best validation candidate, then accepts an additional averaged checkpoint only if the averaged model does not reduce the configured validation selection metric. If every average hurts validation performance, the final greedy soup is the best single Phase 3 checkpoint.

Reported metrics include AUROC, average precision, fixed-threshold F1 at `0.5`, validation-threshold F1, and confusion counts. Dynamic best-threshold F1 is retained only where it is valid for threshold selection on the validation split.

## Seed Aggregation

For article-level reporting, run the same command separately with seeds `1337`, `2022`, `42`, `2026`, and `0`. Each `protocol_results.json` records the seed used for model initialization, dataloading, prompt augmentation, and inference-time style sampling.

Aggregate completed protocol runs:

```bash
python evaluation/aggregate_protocol_seeds.py \
  --root /path/to/runs \
  --output /path/to/seed_aggregate.json \
  --csv-output /path/to/seed_aggregate.csv
```

## Notes

The repository does not include raw CTs, materialized patches, pretrained text encoder weights, or experiment outputs. Those paths are supplied at runtime through the command line.
