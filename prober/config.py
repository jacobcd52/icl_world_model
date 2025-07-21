from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Config:
    """Holds all configurable parameters for data generation, probing, and training."""

    # Model & tokenizer
    model_name: str = "google/gemma-2b-it"  # Default LLM
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Activation extraction
    layer_index: int = -1  # -1 means use final layer
    resid_stream_name: str = "blocks.{layer}.hook_resid_post"  # formatted later

    # Data generation
    n_train_samples: int = 100_000
    n_val_samples: int = 1000
    min_objects: int = 3
    max_objects: int = 5
    world_radius: float = 5.0  # coordinates sampled uniformly from [-radius, radius]
    object_pool: List[str] = field(
        default_factory=lambda: [
            "flower pot",
            "trash can",
            "brick",
            "lamp",
            "chair",
            "table",
            "bicycle",
            "box",
            "cup",
            "book",
            "bottle",
            "umbrella",
            "pencil",
            "notebook",
            "shoe",
            "backpack",
            "dog",
            "cat",
            "bird",
            "fish",
            "horse",
            "sheep",
            "cow",
        ]
    )

    # File paths
    dataset_path: str = "data/object_world.npz"
    probe_weights_path: str = "data/probe_weights.npy"

    # Probe training
    regularization: float = 0.001  # ℓ2 regularization coefficient (λ)
    batch_size: int = 2048  # used for probe fitting (CPU-side)

    # Data generation batching
    generation_batch_size: int = 8  # how many prompts to process in parallel when extracting activations

    def resid_hook_name(self):
        """Return the formatted hook name for the desired residual stream layer."""
        if self.layer_index < 0:
            return f"blocks.{self.layer_index}.hook_resid_post"
        return self.resid_stream_name.format(layer=self.layer_index) 