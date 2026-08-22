# Mask R-CNN

The backbone extracts FPN features. The RPN proposes candidate boxes. ROI box heads classify and refine them; the mask head predicts one binary mask per retained ROI. In `train()` the torchvision model returns five losses. In `eval()` it returns `boxes`, `labels`, `scores`, and mask probabilities. The project replaces both box and mask predictors whenever `num_classes` changes.
