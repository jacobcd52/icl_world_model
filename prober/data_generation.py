from __future__ import annotations

import os
import random
import math
from typing import List, Dict, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import Config
from .model_utils import load_model_and_tokenizer, _resolve_hidden_state_index, _num_layers


# ---------------------- Helper functions ----------------------

def _format_object_list(objs: List[str]) -> str:
    """Return a human-readable list like "a lamp, a chair and a book"."""
    if len(objs) == 1:
        return f"a {objs[0]}"
    if len(objs) == 2:
        return f"a {objs[0]} and a {objs[1]}"
    return ", ".join(f"a {o}" for o in objs[:-1]) + f", and a {objs[-1]}"


def _find_subsequence(sequence: List[int], subseq: List[int]):
    """Return the index of the last element of the first occurrence of subseq in `sequence`.
    Returns None if the subsequence does not occur."""
    n, m = len(sequence), len(subseq)
    for i in range(n - m + 1):
        if sequence[i : i + m] == subseq:
            return i + m - 1
    return None


def _vector_to_sentence(dx: float, dy: float, reference: str) -> str:
    """Convert a vector into a natural language sentence relative to `reference`."""
    distance = round(math.sqrt(dx ** 2 + dy ** 2), 1)
    # Direction description (simple N/E/S/W)
    if abs(dx) > abs(dy):
        direction = "to the right of" if dx > 0 else "to the left of"
    else:
        direction = "in front of" if dy > 0 else "behind"
    base = "us" if reference == "us" else f"the {reference}"
    return f"{distance} meters {direction} {base} is"

# ---------------------- Main generator ----------------------

def generate_and_save_dataset(cfg: Config):
    os.makedirs(os.path.dirname(cfg.dataset_path), exist_ok=True)

    print(f"[Info] Activations will be saved to {cfg.dataset_path}")

    model, tokenizer = load_model_and_tokenizer(cfg)  # ensure weights loaded before loop

    all_X: List[np.ndarray] = []
    all_Y: List[np.ndarray] = []

    total_samples = cfg.n_train_samples + cfg.n_val_samples
    rng = random.Random(42)

    # Batch buffers
    batch_ids: List[List[int]] = []
    batch_positions: List[List[int]] = []
    batch_labels: List[List[Tuple[float, float]]] = []

    def _flush_batch():
        if not batch_ids:
            return

        # Pad batch
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        max_len = max(len(ids) for ids in batch_ids)
        padded = np.full((len(batch_ids), max_len), pad_id, dtype=np.int64)
        for i, ids in enumerate(batch_ids):
            padded[i, : len(ids)] = ids

        tokens_tensor = torch.tensor(padded, device=cfg.device)

        target_idx = _resolve_hidden_state_index(cfg.layer_index, _num_layers(model))

        with torch.no_grad():
            outputs = model(tokens_tensor, output_hidden_states=True)

        resid = outputs.hidden_states[target_idx].detach().cpu()  # (B, seq_len, d_model)
        for i, pos_list in enumerate(batch_positions):
            sample_acts = resid[i, pos_list, :].float().numpy()  # (n_obj_i, d_model)
            all_X.append(sample_acts)
            all_Y.append(np.array(batch_labels[i], dtype=np.float32))

        # Clear batch buffers
        batch_ids.clear()
        batch_positions.clear()
        batch_labels.clear()

    for _ in tqdm(range(total_samples), desc="Samples"):
        num_obj = rng.randint(cfg.min_objects, cfg.max_objects)
        objects = rng.sample(cfg.object_pool, num_obj)

        # Assign random coordinates in 2D square
        coords = {obj: (rng.uniform(-cfg.world_radius, cfg.world_radius), rng.uniform(-cfg.world_radius, cfg.world_radius)) for obj in objects}

        # Build user prompt with relational sentences
        sentences = []
        for idx, obj in enumerate(objects):
            if idx == 0:
                x, y = coords[obj]
                sentence = _vector_to_sentence(x, y, "us") + f" a {obj}."
            else:
                # reference previous object for variability
                ref_obj = objects[rng.randint(0, idx - 1)]
                x1, y1 = coords[obj]
                x0, y0 = coords[ref_obj]
                sentence = _vector_to_sentence(x1 - x0, y1 - y0, ref_obj) + f" a {obj}."
            sentences.append(sentence)
        user_prompt = " ".join(sentences)

        # Assistant response with randomised object order
        resp_order = objects.copy()
        rng.shuffle(resp_order)
        assistant_resp = f"The user has asked me about {_format_object_list(resp_order)}."

        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_resp},
        ]

        token_ids: List[int] = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        token_list = token_ids  # alias

        # Find positions of each object's final token
        positions: List[int] = []
        sample_labels: List[Tuple[float, float]] = []
        for obj in objects:
            token_variants = [
                tokenizer.encode(obj, add_special_tokens=False),
                tokenizer.encode(" " + obj, add_special_tokens=False),
            ]
            pos = None
            for variant in token_variants:
                if variant:
                    pos = _find_subsequence(token_list, variant)
                    if pos is not None:
                        break
            if pos is None:
                raise ValueError(f"Could not find token sequence for object '{obj}' in encoded prompt.")
            positions.append(pos)
            sample_labels.append(coords[obj])

        # Accumulate in batch buffers
        batch_ids.append(token_ids)
        batch_positions.append(positions)
        batch_labels.append(sample_labels)

        if len(batch_ids) >= cfg.generation_batch_size:
            _flush_batch()

    # Flush remaining
    _flush_batch()

    X = np.concatenate(all_X, axis=0)  # (N_total_objects, d_model)
    Y = np.concatenate(all_Y, axis=0)  # (N_total_objects, 2)

    np.savez(cfg.dataset_path, X=X, Y=Y)
    print(f"Saved dataset to {cfg.dataset_path}. Shapes: X={X.shape}, Y={Y.shape}") 