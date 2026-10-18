"""Reproducibility, configuration, and model environment helpers."""
from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

def save_json(obj, path):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def load_config(path="configs/config.yaml"):
    import yaml
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)

