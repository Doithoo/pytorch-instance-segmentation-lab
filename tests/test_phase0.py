from __future__ import annotations

from instance_segmenter import __version__
from instance_segmenter.cli import main


def test_version_is_exposed() -> None:
    assert __version__ == "0.1.0"


def test_cli_accepts_empty_arguments() -> None:
    assert main([]) == 0
