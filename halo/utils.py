"""Misc utilities: configs, logging, deterministic seeding."""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def get_logger(name: str = "halo", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s %(levelname)s — %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


def seed_everything(seed: int = 0) -> None:
    """Best-effort determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def dump_json(obj: Any, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, ensure_ascii=False)


def load_yaml(path: str | os.PathLike) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def write_manifest(out_dir: str | os.PathLike, **fields: Any) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    dump_json(fields, path)
    return str(path)
