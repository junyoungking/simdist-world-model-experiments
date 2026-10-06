"""Path and import routing for the public experiment/evidence subset.

This module supplies paths, not replacements for SimDist or IsaacLab APIs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PORTFOLIO_ROOT = Path(__file__).resolve().parent


def simdist_root() -> Path:
    value = os.environ.get("SIMDIST_ROOT")
    if not value:
        raise RuntimeError("Set SIMDIST_ROOT to a separate pinned SimDist checkout; see docs/REPRODUCIBILITY.md")
    root = Path(value).expanduser().resolve()
    if not (root / "simdist").is_dir() or not (root / "config").is_dir():
        raise FileNotFoundError("SIMDIST_ROOT must contain simdist/ and config/")
    return root


def checkpoint_root() -> Path:
    value = os.environ.get("SIMDIST_CHECKPOINT_ROOT")
    return Path(value).expanduser().resolve() if value else simdist_root() / "checkpoints"


def artifact_root() -> Path:
    value = os.environ.get("SIMDIST_ARTIFACT_ROOT")
    return Path(value).expanduser().resolve() if value else PORTFOLIO_ROOT / "results"


def artifact_dir(family: str) -> Path:
    return artifact_root() / family


def result_path(relative: str) -> Path:
    """Prefer supplied full-run evidence, with public summaries as a fallback."""
    external = artifact_root() / relative
    return external if external.exists() else PORTFOLIO_ROOT / "results" / relative


def legacy_dir(family: str) -> Path:
    aliases = {"10_t1_data_amount_control": "t1", "11_t3_support_causal": "t3",
               "12_t2_v2_common_control_pairwise": "t2",
               "mass8_privileged_ppo_convergence_20260929T072901Z": "privileged_ppo"}
    return artifact_dir(aliases[family]) if family in aliases else artifact_root() / "legacy" / family


def document_path(family: str, filename: str) -> Path:
    """Read original external protocol when supplied, else the public document.

    Historical hash verification requires original, unmodified external files.
    """
    original = artifact_dir(family) / "00_protocol" / filename
    if original.is_file():
        return original
    return PORTFOLIO_ROOT / "docs" / family / filename


def resolve_artifact_path(value) -> Path:
    """Resolve public metadata identifiers or an ordinary CLI filesystem path."""
    if isinstance(value, Path):
        return value
    value = os.fspath(value)
    if value.startswith("artifact://"):
        return artifact_root() / value.removeprefix("artifact://")
    if value.startswith("checkpoint://"):
        return checkpoint_root() / value.removeprefix("checkpoint://")
    if value.startswith("portfolio://"):
        return PORTFOLIO_ROOT / value.removeprefix("portfolio://")
    if value.startswith("simdist://"):
        return simdist_root() / value.removeprefix("simdist://")
    return Path(value)


def configure_imports() -> None:
    """Connect real external packages and selected local helper modules."""
    paths = [PORTFOLIO_ROOT, PORTFOLIO_ROOT / "experiments",
             PORTFOLIO_ROOT / "experiments/simdist_c",
             PORTFOLIO_ROOT / "experiments/extended_adaptation_v1",
             PORTFOLIO_ROOT / "experiments/candidate_rank_v1"]
    if os.environ.get("SIMDIST_ROOT"):
        paths.append(simdist_root())
    for path in reversed(paths):
        item = str(path)
        if item not in sys.path:
            sys.path.insert(0, item)
