"""Generate fixed, auditable Penn-Fudan manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from instance_segmenter.data.manifest import prepare_penn_fudan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare fixed Penn-Fudan manifests")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    args = parser.parse_args(argv)
    metadata = prepare_penn_fudan(args.data_dir, args.manifest_dir)
    print(f"identity={metadata.identity}")
    print(" ".join(f"{split}={count}" for split, count in metadata.split_counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
