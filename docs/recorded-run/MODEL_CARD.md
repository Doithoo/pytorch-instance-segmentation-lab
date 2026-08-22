# Protocol-v2 Mask R-CNN Model Card

## Model

- Architecture: torchvision `maskrcnn_resnet50_fpn`
- Initialization: `MaskRCNN_ResNet50_FPN_Weights.COCO_V1`, prediction heads replaced for background/person
- Dataset: Penn-Fudan Pedestrian, source-stratified manifests, 136 train / 17 valid / 17 test
- Training: 20 epochs, SGD, StepLR, horizontal flip, CUDA AMP, seed 42
- Selection: epoch 10 by validation mask AP `0.766694`
- Runtime: Kaggle Tesla T4, kernel version 2

## Evaluation

Protocol v2 retains the complete confidence ranking (`metric_score_floor=0.0`) and uses mask threshold 0.5. On the fixed held-out test split:

| Metric | Value |
|---|---:|
| Mask AP / AP50 / AP75 | 0.756093 / 1.000000 / 0.855337 |
| Box AP / AP50 / AP75 | 0.846439 / 1.000000 / 0.935175 |
| Mask AR@100 / Box AR@100 | 0.782500 / 0.872500 |

The test set contains only 17 images and 40 targets. These values are a reproducible teaching baseline, not evidence of production-level generalization.

## Intended Use

The checkpoint is intended for learning, reproducibility exercises, Penn-Fudan inference, transfer-learning demonstrations, and regression testing. It predicts the single foreground class `person`.

It is not intended for safety-critical decisions, surveillance deployment, demographic performance claims, or direct use on unrelated domains without new validation. Scores are not calibrated probabilities, and the small dataset does not support broad fairness or robustness conclusions.

## Artifact and Security

- Local ignored artifact: `artifacts/protocol-v2-reference/best.pt`
- Kaggle output: `artifacts/reference-maskrcnn/best.pt` in kernel version 2
- SHA-256: `1c28ed12b3b9d380bb3888a99267c6d7694efcfd2a93b22867abbb2ccb6b3d57`
- Size: 334.8 MB

PyTorch `.pt` files are pickle-based trusted inputs. Verify the SHA-256 and load only this trusted artifact.

```bash
uv run instance-segment predict \
  --checkpoint artifacts/protocol-v2-reference/best.pt \
  --image path/to/image.png \
  --output artifacts/prediction
```

## Provenance

See [`run.yaml`](run.yaml), [`config.yaml`](config.yaml), [`environment.json`](environment.json), [`metrics.csv`](metrics.csv), and the exact submitted [`run_kaggle-v2.py`](kaggle/run_kaggle-v2.py). Code is MIT-licensed; Penn-Fudan and upstream pretrained weights retain their respective terms.
