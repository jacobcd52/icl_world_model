#!/usr/bin/env python
"""Fit linear probe on previously generated activations.

Usage:
    python scripts/train_probe.py [--config path/to/override.yaml]

The probe weights are saved to cfg.probe_weights_path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

# Ensure project root in path for module resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # type: ignore

from prober import Config, train_probe


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
    parser = argparse.ArgumentParser(description="Train linear probe on saved dataset.")
    parser.add_argument("--config", type=str, default=None, help="YAML file to override Config defaults.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Training probe with config:\n", cfg)
    train_probe(cfg)


if __name__ == "__main__":
    main() 