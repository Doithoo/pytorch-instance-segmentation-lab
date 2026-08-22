"""Load a checkpoint and write thresholded masks plus an overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

from instance_segmenter.inference.predictor import Predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = Predictor.from_checkpoint(args.checkpoint, device="cpu").predict_single(args.image, args.output)
    print(result.overlay_path)


if __name__ == "__main__":
    main()
