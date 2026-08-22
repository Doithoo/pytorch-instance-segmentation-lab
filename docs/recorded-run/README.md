# Kaggle Protocol-v2 Reference Run

[Chinese](README.zh-CN.md) | [Kaggle kernel version 2](https://www.kaggle.com/code/yashowhoo/pytorch-instance-segmentation-lab-penn-fudan-gpu) | [Workflow](../guides/kaggle.md) | [ADR 0002](../architecture/0002-evaluation-and-splits.md)

Status: `COMPLETE_PROTOCOL_V2`

The 20-epoch replacement run completed on a Kaggle Tesla T4 with the source-stratified manifests and full confidence-ranked COCO AP. Epoch 10 was selected only by validation `mask_map`; the fixed 17-image test split was evaluated once afterwards.

| Protocol-v2 item | Result |
|---|---:|
| Best validation mask AP | **0.766694** |
| Test mask AP / AP50 / AP75 | **0.756093** / 1.000000 / 0.855337 |
| Test bbox AP / AP50 / AP75 | **0.846439** / 1.000000 / 0.935175 |
| Test mask AR@100 / bbox AR@100 | 0.782500 / 0.872500 |
| Test images / targets / metric predictions | 17 / 40 / 54 |
| Analysis predictions / FP / FN at score 0.5 | 45 / 5 / 0 |
| Training / evaluation / total | 537.431s / 4.609s / 585.735s |

![Source-stratified Penn-Fudan preview](assets/dataset-preview.png)

## Ranked error analysis

The evaluator retained all predictions for AP and separately used score 0.5 for readable error analysis. The highest-ranked case contains six matched targets and three additional predictions; all 40 test targets were matched at the analysis threshold.

![Worst-case protocol-v2 prediction](evaluation/visualizations/worst-01-96415228031564514-prediction.png)

A real single-image prediction, its JSON records, and binary masks are available under [`predictions/FudanPed00028/`](predictions/FudanPed00028/instances.json).

## Auditable artifacts

- [`run.yaml`](run.yaml): curated protocol, environment, timings, metrics, identities, and hashes.
- [`MODEL_CARD.md`](MODEL_CARD.md): intended use, limitations, checkpoint location, and security guidance.
- [`config.yaml`](config.yaml): exact resolved Kaggle paths, CUDA AMP, and thresholds.
- [`environment.json`](environment.json), [`events.jsonl`](events.jsonl), and [`metrics.csv`](metrics.csv): structured provenance and all 20 epochs.
- [`kaggle-run-summary.json`](kaggle-run-summary.json): unrounded runner output.
- [`evaluation/`](evaluation/evaluation.json): JSON, per-class/per-image CSV, and four ranked ground-truth/prediction pairs.
- [`kaggle/run_kaggle-v2.py`](kaggle/run_kaggle-v2.py): exact submitted version-2 runner; embedded archive SHA-256 `41fa5e...b24a`.
- [`kaggle/run_kaggle.py`](kaggle/run_kaggle.py): current generated runner for future submissions.
- [`legacy-v1/`](legacy-v1/README.md): preserved superseded score-filtered run and reports.

The 334.8 MB `best.pt` and `last.pt` remain in the private Kaggle kernel output. Verified SHA-256 values are `1c28ed12...b3d57` and `bda01afe...2ad1`. Load them only as trusted checkpoints.
