# Metrics

`mask_map` is COCO-style AP averaged from mask IoU 0.50 to 0.95 and selects `best.pt`. `mask_map_50`, `mask_map_75`, and `mask_mar_100` add detail. `bbox_*` reports the equivalent box metrics. Pixel accuracy is intentionally not a project metric because background dominance hides failed instance contours.
