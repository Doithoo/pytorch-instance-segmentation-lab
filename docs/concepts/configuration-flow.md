# Configuration Flow

Defaults are dataclasses. YAML may override only known fields. `--set key value` overrides YAML. The resolved config is saved in the run directory and checkpoint; Kaggle replaces only device, paths, workers, and AMP in its recorded resolved config.
