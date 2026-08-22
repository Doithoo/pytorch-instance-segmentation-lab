# Troubleshooting

- `CUDA was requested`: run `instance-segment doctor --device auto`, use a CPU dry-run, or request T4/newer in Kaggle.
- `manifest hash mismatch`: do not edit CSV manually. Restore committed manifests or rerun the appropriate prepare command and start a new experiment.
- `resume configuration changes immutable fields`: resume the original trajectory with only epochs/device/workers changed, or choose a new run without `--resume`.
- `metrics schema is incompatible`: keep the historical directory unchanged and resume into a new run directory; do not append across protocol versions.
- `evaluation output already exists`: choose a new directory or pass `--overwrite` deliberately.
- `unknown config field`: inspect `instance-segment show-config --config ...`; YAML is strict.
- COCO decode failure: validate polygon/RLE structure, dimensions, category consistency, and that all paths stay under the data root.
- Out of memory: reduce batch size/min-max image size, use MobileNetV3, enable CUDA AMP, or lower worker count. Record the altered config when comparing runs.
