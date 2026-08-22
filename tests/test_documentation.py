from __future__ import annotations

import csv
import hashlib
import json
import re
import runpy
import shlex
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml

from instance_segmenter.cli import build_parser
from instance_segmenter.models.registry import list_models

ROOT = Path(__file__).parents[1]
PUBLICATION_ROOTS = [ROOT / name for name in ("docs", "configs", "examples", "scripts", "tests", "src")]
ROOT_PAIRS = ("README", "CONTRIBUTING")
WORKFLOW = "download -> prepare -> verify -> inspect -> dry-run -> train -> evaluate -> compare/predict"
WORKFLOW_ZH = "下载 -> 准备 -> 校验 -> 检查 -> dry-run -> 训练 -> 评估 -> 对比/推理"


def _publication_pages() -> list[Path]:
    pages = [ROOT / "README.md", ROOT / "README.zh-CN.md", ROOT / "CONTRIBUTING.md", ROOT / "CONTRIBUTING.zh-CN.md"]
    for directory in PUBLICATION_ROOTS:
        pages.extend(directory.rglob("*.md"))
    return sorted(set(pages))


def _broken_local_links(pages: list[Path]) -> list[str]:
    missing = []
    for source in pages:
        for raw_target in re.findall(r"!?\[[^]]*]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            target = unquote(raw_target.split()[0]).split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (source.parent / target).resolve().exists():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
    return missing


def test_english_and_chinese_pages_exist_in_pairs() -> None:
    missing = []
    for stem in ROOT_PAIRS:
        for path in (ROOT / f"{stem}.md", ROOT / f"{stem}.zh-CN.md"):
            if not path.is_file():
                missing.append(path.relative_to(ROOT).as_posix())
    for directory in PUBLICATION_ROOTS:
        for chinese in directory.rglob("*.zh-CN.md"):
            english = chinese.with_name(chinese.name.replace(".zh-CN.md", ".md"))
            if not english.is_file():
                missing.append(english.relative_to(ROOT).as_posix())
        for english in directory.rglob("*.md"):
            if english.name.endswith(".zh-CN.md"):
                continue
            chinese = english.with_name(english.name.removesuffix(".md") + ".zh-CN.md")
            if not chinese.is_file():
                missing.append(chinese.relative_to(ROOT).as_posix())
    assert not missing, "missing language page(s):\n" + "\n".join(sorted(set(missing)))


def test_all_local_markdown_links_resolve() -> None:
    missing = _broken_local_links(_publication_pages())
    assert not missing, "broken local links:\n" + "\n".join(missing)


def test_readmes_publish_workflow_and_recorded_metric() -> None:
    for path in (ROOT / "README.md", ROOT / "README.zh-CN.md"):
        content = path.read_text(encoding="utf-8")
        assert (WORKFLOW_ZH if path.name.endswith(".zh-CN.md") else WORKFLOW) in content
        assert "0.756093" in content
        assert "docs/recorded-run/" in content
        assert "prepare-data --data-dir" not in content


def test_recorded_run_index_is_current() -> None:
    for path in (ROOT / "docs/recorded-run/README.md", ROOT / "docs/recorded-run/README.zh-CN.md"):
        content = path.read_text(encoding="utf-8")
        assert "0.756093" in content
        assert "does not yet" not in content
        assert "目前不声明" not in content


def test_documented_cli_lines_use_real_parser_options() -> None:
    parser = build_parser()
    failures = []
    for source in _publication_pages():
        for line in source.read_text(encoding="utf-8").splitlines():
            command = line.strip()
            if command.endswith("\\"):
                continue
            if command.startswith("uv run instance-segment "):
                command = command.removeprefix("uv run ")
            elif not command.startswith("instance-segment "):
                continue
            try:
                parser.parse_args(shlex.split(command)[1:])
            except SystemExit as exc:
                if exc.code != 0:
                    failures.append(f"{source.relative_to(ROOT)}: {command}")
    assert not failures, "invalid documented CLI command(s):\n" + "\n".join(failures)


def test_documented_python_entry_points_and_configs_exist() -> None:
    missing = []
    for source in _publication_pages():
        content = source.read_text(encoding="utf-8")
        for target in re.findall(r"uv run(?: --extra \w+)? python\s+((?:scripts|examples)/[^\s`\\]+)", content):
            if "<" in target or target == "...":
                continue
            if not (ROOT / target).is_file():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
        for target in re.findall(r"--config\s+([^\s`\\]+)", content):
            if target == "PATH" or "<" in target or not target.startswith(("configs/", "data/", "artifacts/")):
                continue
            if not (ROOT / target).is_file():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
    assert not missing, "missing documented path(s):\n" + "\n".join(missing)


def test_model_catalog_lists_every_registered_model() -> None:
    for suffix in ("", ".zh-CN"):
        content = (ROOT / f"docs/reference/model-zoo{suffix}.md").read_text(encoding="utf-8")
        documented = {name for name in list_models() if re.search(rf"\|\s*`{re.escape(name)}`\s*\|", content)}
        assert documented == set(list_models())


def test_protocol_v2_recorded_run_is_internally_consistent() -> None:
    recorded = ROOT / "docs" / "recorded-run"
    summary = json.loads((recorded / "kaggle-run-summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads((recorded / "evaluation" / "evaluation.json").read_text(encoding="utf-8"))
    run = yaml.safe_load((recorded / "run.yaml").read_text(encoding="utf-8"))
    with (recorded / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["status"] == "complete"
    assert summary["completed_epochs"] == len(rows) == 20
    assert summary["dataset_identity"] == run["dataset_identity"] == evaluation["dataset_identity"]
    assert summary["metric_score_floor"] == evaluation["metric_score_floor"] == 0.0
    assert summary["metric_protocol"] == evaluation["metric_protocol"]
    assert summary["metrics"] == evaluation["metrics"]
    assert max(float(row["valid_mask_map"]) for row in rows) == pytest.approx(summary["best_valid_mask_map"])
    assert run["best_checkpoint_sha256"] == summary["checkpoint_sha256"]

    submitted_path = recorded / "kaggle" / "run_kaggle-v2.py"
    submitted = runpy.run_path(str(submitted_path))
    assert hashlib.sha256(submitted_path.read_bytes()).hexdigest() == run["submitted_runner_sha256"]
    assert submitted["PROJECT_ARCHIVE_SHA256"] == summary["source_archive_sha256"]
    assert submitted["PROJECT_ARCHIVE_BYTES"] == summary["source_archive_bytes"]
