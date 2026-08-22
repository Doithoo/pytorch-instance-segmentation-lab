# Checkpoint Schema

Format version 1 stores model and optimizer state, scheduler state, completed epoch, best validation metric/epoch, label schema, resolved config, manifest hashes, Python/Torch versions, and RNG state. Loading rejects an incompatible format, model name, label schema, or tensor state shape.
