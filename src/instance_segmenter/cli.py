from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from instance_segmenter import __version__
from instance_segmenter.config import config_to_dict, load_config, load_config_with_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="instance-segment", description="PyTorch instance segmentation learning lab")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    show_config = subparsers.add_parser("show-config", help="show resolved YAML configuration")
    _add_config_arguments(show_config)
    show_config.set_defaults(handler=_show_config)

    init_config = subparsers.add_parser("init-config", help="copy an installed configuration template")
    init_config.add_argument("name", nargs="?", default="learning_minimal")
    init_config.add_argument("--list", action="store_true", dest="list_templates")
    init_config.add_argument("--output", type=Path)
    init_config.add_argument("--overwrite", action="store_true")
    init_config.set_defaults(handler=_init_config)

    doctor = subparsers.add_parser("doctor", help="inspect the selected compute device")
    doctor.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    doctor.set_defaults(handler=_doctor)

    prepare = subparsers.add_parser("prepare-data", help="write fixed Penn-Fudan manifests")
    _add_data_paths(prepare)
    prepare.set_defaults(handler=_prepare_data)

    prepare_coco = subparsers.add_parser("prepare-coco", help="prepare manifests from COCO instance JSON")
    _add_data_paths(prepare_coco)
    prepare_coco.add_argument("--train-annotations", type=Path, required=True)
    prepare_coco.add_argument("--valid-annotations", type=Path, required=True)
    prepare_coco.add_argument("--test-annotations", type=Path, required=True)
    prepare_coco.set_defaults(handler=_prepare_coco)

    verify = subparsers.add_parser("verify-data", help="verify source data and fixed manifests")
    _add_data_paths(verify)
    verify.set_defaults(handler=_verify_data)

    inspect = subparsers.add_parser("inspect-data", help="summarize one prepared split")
    _add_data_paths(inspect)
    inspect.add_argument("--split", choices=("train", "valid", "test"), default="train")
    inspect.set_defaults(handler=_inspect_data)

    list_registered_datasets = subparsers.add_parser("list-datasets", help="list built-in dataset providers")
    list_registered_datasets.set_defaults(handler=_list_datasets)

    list_registered_models = subparsers.add_parser("list-models", help="list registered instance segmentation models")
    list_registered_models.set_defaults(handler=_list_models)

    model_info = subparsers.add_parser("model-info", help="show one model's teaching metadata")
    model_info.add_argument("name")
    model_info.set_defaults(handler=_model_info)

    train = subparsers.add_parser("train", help="train an instance segmentation model")
    _add_config_arguments(train)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--resume", type=Path)
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on one fixed split")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "valid", "test"), default="test")
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    evaluate.add_argument("--metric-score-floor", type=_probability)
    evaluate.add_argument("--score-threshold", type=_probability)
    evaluate.add_argument("--mask-threshold", type=_probability)
    evaluate.add_argument("--plot", action="store_true")
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.set_defaults(handler=_evaluate)

    predict = subparsers.add_parser("predict", help="predict independent instances for one image")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    predict.add_argument("--score-threshold", type=_probability)
    predict.add_argument("--mask-threshold", type=_probability)
    predict.add_argument("--overwrite", action="store_true")
    predict.set_defaults(handler=_predict)

    compare = subparsers.add_parser("compare-runs", help="rank compatible completed runs")
    compare.add_argument("run_dirs", nargs="+", type=Path)
    compare.add_argument("--metric", default="valid_mask_map")
    compare.add_argument("--allow-incompatible", action="store_true")
    compare.set_defaults(handler=_compare_runs)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", nargs=2, default=[], metavar=("KEY", "VALUE"))


def _add_data_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))


def _show_config(args: argparse.Namespace) -> int:
    config, sources = load_config_with_sources(args.config, [tuple(item) for item in args.overrides])
    resolved = config_to_dict(config)
    resolved["sources"] = sources
    print(yaml.safe_dump(resolved, sort_keys=False), end="")
    return 0


def _init_config(args: argparse.Namespace) -> int:
    from instance_segmenter.resources import CONFIG_TEMPLATES, copy_config_template

    if args.list_templates:
        print("\n".join(CONFIG_TEMPLATES))
        return 0
    output = args.output if args.output is not None else Path(f"{args.name}.yaml")
    print(copy_config_template(args.name, output, overwrite=args.overwrite))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    from instance_segmenter.preflight import inspect_device

    print(yaml.safe_dump(asdict(inspect_device(args.device)), sort_keys=False), end="")
    return 0


def _prepare_data(args: argparse.Namespace) -> int:
    from instance_segmenter.data.manifest import prepare_penn_fudan

    metadata = prepare_penn_fudan(args.data_dir, args.manifest_dir)
    print(f"identity={metadata.identity}")
    print(" ".join(f"{split}={count}" for split, count in metadata.split_counts.items()))
    return 0


def _prepare_coco(args: argparse.Namespace) -> int:
    from instance_segmenter.data.coco import prepare_coco_instances

    metadata = prepare_coco_instances(
        args.data_dir,
        args.manifest_dir,
        {
            "train": args.train_annotations,
            "valid": args.valid_annotations,
            "test": args.test_annotations,
        },
    )
    print(f"identity={metadata.identity}")
    print(" ".join(f"{split}={count}" for split, count in metadata.split_counts.items()))
    return 0


def _verify_data(args: argparse.Namespace) -> int:
    from instance_segmenter.data.manifest import verify_prepared_data

    metadata = verify_prepared_data(args.data_dir, args.manifest_dir)
    print(f"verified identity={metadata.identity}")
    return 0


def _inspect_data(args: argparse.Namespace) -> int:
    from instance_segmenter.data.inspection import inspect_prepared_data

    print(yaml.safe_dump(inspect_prepared_data(args.data_dir, args.manifest_dir, args.split), sort_keys=False), end="")
    return 0


def _list_datasets(_args: argparse.Namespace) -> int:
    from instance_segmenter.data.registry import get_dataset_spec, list_datasets

    print("name\tdescription")
    for name in list_datasets():
        print(f"{name}\t{get_dataset_spec(name).description}")
    return 0


def _list_models(_args: argparse.Namespace) -> int:
    from instance_segmenter.models.registry import get_model_spec, list_models

    print("name\tweights\tdescription")
    for name in list_models():
        spec = get_model_spec(name)
        print(f"{name}\t{','.join(spec.supported_weights)}\t{spec.description}")
    return 0


def _model_info(args: argparse.Namespace) -> int:
    from instance_segmenter.models.registry import get_model_spec

    spec = get_model_spec(args.name)
    print(f"name: {spec.name}")
    print(f"description: {spec.description}")
    print(f"weights: {', '.join(spec.supported_weights)}")
    print("input_notes:")
    for note in spec.input_notes:
        print(f"  - {note}")
    return 0


def _train(args: argparse.Namespace) -> int:
    from instance_segmenter.training.train import run_training

    config = load_config(args.config, [tuple(item) for item in args.overrides])
    if args.device is not None:
        config = replace(config, device=args.device)
    result = run_training(config, resume=args.resume, dry_run_mode=args.dry_run)
    if result.dry_run_result is not None:
        diagnostics = result.dry_run_result
        print(f"image_shapes={diagnostics.image_shapes}")
        print(f"target_counts={diagnostics.target_counts}")
        for name, value in diagnostics.losses.items():
            print(f"{name}={value}")
        print("dry-run OK")
    else:
        print(result.run_dir)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from instance_segmenter.evaluation.evaluate import evaluate_checkpoint

    result = evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
        metric_score_floor=args.metric_score_floor,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        plot=args.plot,
        overwrite=args.overwrite,
    )
    print(result.output_dir)
    return 0


def _predict(args: argparse.Namespace) -> int:
    from instance_segmenter.inference.predictor import Predictor

    result = Predictor.from_checkpoint(args.checkpoint, device=args.device).predict_single(
        args.image,
        args.output,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        overwrite=args.overwrite,
    )
    print(result.output_dir)
    return 0


def _compare_runs(args: argparse.Namespace) -> int:
    from instance_segmenter.evaluation.comparison import compare_runs

    results = compare_runs(args.run_dirs, args.metric, allow_incompatible=args.allow_incompatible)
    print("run\tmetric\tvalue\tepoch")
    for result in results:
        epoch = "evaluation" if result.epoch is None else str(result.epoch)
        print(f"{result.run_dir}\t{result.metric}\t{result.value:.6f}\t{epoch}")
    return 0


def _probability(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return parsed
