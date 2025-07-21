#!/usr/bin/env python
"""Generate activation+label dataset for linear probing.

Usage:
    python scripts/collect_dataset.py [--config path/to/override.yaml]

The optional YAML file may contain any attributes from prober.Config to override.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import os
import sys

# Guarantee project root is on PYTHONPATH when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # type: ignore

from prober import Config, generate_and_save_dataset


def load_config(yaml_path: str | None) -> Config:
    cfg = Config()
    if yaml_path is None:
        return cfg

    yaml_data = yaml.safe_load(Path(yaml_path).read_text()) or {}
    for key, value in yaml_data.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Config has no field '{key}'")
        setattr(cfg, key, value)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Generate dataset for spatial probing.")
    parser.add_argument("--config", type=str, default=None, help="YAML file to override Config defaults.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Running dataset generation with config:\n", cfg)
    generate_and_save_dataset(cfg)


if __name__ == "__main__":
    main() 