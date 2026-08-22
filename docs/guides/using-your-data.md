# Using Your Data

Use matching image and instance-ID mask files. Keep 0 for background and assign every object a different positive integer. Implement `module:build_dataset` with the keyword arguments shown in `examples/extensions/my_dataset.py`, return image/target pairs with the documented dtypes, then set `data.factory` in YAML. Test horizontal flip and resize before long training.
