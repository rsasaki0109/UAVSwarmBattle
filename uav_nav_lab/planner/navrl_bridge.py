"""Lazy loader for the upstream NavRL quick-demo policy (Zhefan-Xu/NavRL)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def default_navrl_root() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "NavRL"


def ensure_navrl_path(navrl_root: str | Path) -> Path:
    root = Path(navrl_root).resolve()
    quick = root / "quick-demos"
    if not quick.is_dir():
        raise FileNotFoundError(
            f"NavRL quick-demos not found at {quick}. "
            "Run: bash scripts/setup_navrl_adapter.sh"
        )
    qstr = str(quick)
    if qstr not in sys.path:
        sys.path.insert(0, qstr)
    return root


def _patch_torchrl_compat() -> None:
    """NavRL quick-demos target an older torchrl API."""
    import torchrl.data as td

    if not hasattr(td, "CompositeSpec"):
        td.CompositeSpec = td.Composite  # type: ignore[attr-defined]
    if not hasattr(td, "UnboundedContinuousTensorSpec"):
        td.UnboundedContinuousTensorSpec = td.UnboundedContinuous  # type: ignore[attr-defined]


def load_agent(navrl_root: str | Path, *, device: str = "cpu") -> Any:
    ensure_navrl_path(navrl_root)
    try:
        import torch
        _patch_torchrl_compat()
        from agent import Agent  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "NavRL requires torch + torchrl + tensordict. "
            "Install: pip install -e '.[navrl]'"
        ) from exc
    dev = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    return Agent(device=dev)


def load_utils(navrl_root: str | Path) -> Any:
    ensure_navrl_path(navrl_root)
    _patch_torchrl_compat()
    import utils  # type: ignore[import-not-found]

    return utils
