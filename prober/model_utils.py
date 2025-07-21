from __future__ import annotations

from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import Config

# Cache loaded models to avoid duplicate memory usage
_MODEL_CACHE: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}


def _num_layers(model: AutoModelForCausalLM) -> int:
    """Return the number of transformer blocks for a HF model, trying common field names."""
    cfg = model.config
    for attr in ("n_layer", "num_hidden_layers", "num_layers"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise AttributeError("Unable to determine number of layers from model config")


def _resolve_hidden_state_index(layer_index: int, n_layers: int) -> int:
    """Convert a user-specified layer index (can be negative, 0=first block) into the
    index used in the `hidden_states` tuple returned by HF models. hidden_states[0]
    corresponds to the embedding output, so we offset by +1."""
    if layer_index < 0:
        layer_index = n_layers + layer_index  # -1 => last block
    if layer_index < 0 or layer_index >= n_layers:
        raise ValueError(f"layer_index {layer_index} out of range (n_layers={n_layers})")
    return layer_index + 1  # +1 to skip embeddings


def load_model_and_tokenizer(cfg: Config):
    if cfg.model_name in _MODEL_CACHE:
        return _MODEL_CACHE[cfg.model_name]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer for this model does not provide a chat template.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # load on CPU to avoid device-mix issues then move
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(cfg.device)
    model.eval()

    _MODEL_CACHE[cfg.model_name] = (model, tokenizer)
    return model, tokenizer


def tokenize_chat(messages: List[Dict[str, str]], tokenizer, device):
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).to(device)
    return tokens


def extract_residual_activations(
    cfg: Config, tokens: torch.Tensor, token_positions: List[int]
) -> torch.Tensor:
    """Return residual activations at specified positions from the configured layer."""
    model, _ = load_model_and_tokenizer(cfg)
    target_idx = _resolve_hidden_state_index(cfg.layer_index, _num_layers(model))

    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    resid = outputs.hidden_states[target_idx].squeeze(0)  # (seq_len, d_model)
    gathered = resid[token_positions].cpu().float()
    return gathered 