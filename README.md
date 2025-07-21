# Linear Probing on Language Models

This repository provides a minimal, **concise** pipeline for training linear probes on residual stream activations of a language model.

* **Default model**: `google/gemma-2b-it` (instruct-tuned, chat template friendly).
* **Probing task**: Reconstruct 2-D coordinates of objects mentioned by the assistant.
* **Frameworks**: [Transformers](https://github.com/huggingface/transformers).

## Quickstart

```bash
# 1. Install deps (torch CUDA wheel may vary)
pip install -r requirements.txt

# 2. Generate dataset (≈ 10–15 min on GPU)
python scripts/collect_dataset.py          # uses default config

# 3. Train probe (fast, CPU-only)
python scripts/train_probe.py              # requires dataset file
```

Pass a YAML file to override any `Config` field:

```bash
python scripts/collect_dataset.py --config my_cfg.yaml
python scripts/train_probe.py   --config my_cfg.yaml
```

## Configuration
All tunable options live in `prober/config.py` and are exposed via the `Config` dataclass. Key fields:

* `model_name` – HF repo to load
* `layer_index` – which residual stream layer to extract (`-1`=final)
* `regularization` – ℓ2 penalty λ for the probe
* `n_train_samples`, `n_val_samples` – dataset size

Edit these values in one place, rerun generation/training.

## File Structure
```
prober/
  ├── __init__.py          # thin public API
  ├── config.py            # all hyper-params
  ├── data_generation.py   # creates dataset & saves to .npz
  ├── model_utils.py       # model load + activation hooks
  └── probe_training.py    # OLS probe fitting
requirements.txt
README.md
```

## Notes
* Dataset cached at `data/object_world.npz` (configurable).
* Probe weights saved as `data/probe_weights.npy`.
* Code intentionally minimal—no catch-all try/except blocks. Errors propagate.

Enjoy probing! :)
