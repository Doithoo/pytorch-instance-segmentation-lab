# Mask R-CNN Flow

`list[CHW image]` enters torchvision's transform and ResNet-50 FPN. RPN proposes regions, ROI heads classify/refine boxes, and the mask head produces one probability map per kept instance. Training consumes target lists and returns losses; inference consumes only images and returns variable-length predictions.
