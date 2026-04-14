from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download  # type: ignore[reportMissingImports]


DEFAULT_INTERPLM_REPO_ID = "Elana/InterPLM-esm2-8m"


class InterPLMSparseAutoencoder(nn.Module):
    """Minimal InterPLM sparse autoencoder used for feature extraction."""

    def __init__(self, input_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.register_parameter("bias", nn.Parameter(torch.zeros(self.input_dim)))
        self.encoder = nn.Linear(self.input_dim, self.feature_dim)
        self.decoder = nn.Linear(self.feature_dim, self.input_dim, bias=False)

    def encode(self, residue_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(residue_embeddings - self.bias))

    def forward(self, residue_embeddings: torch.Tensor) -> torch.Tensor:
        activations = self.encode(residue_embeddings)
        return self.decoder(activations) + self.bias


def load_interplm_sae(
    plm_layer: int,
    repo_id: str = DEFAULT_INTERPLM_REPO_ID,
    normalized: bool = True,
    cache_dir: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> InterPLMSparseAutoencoder:
    layer_dir = f"layer_{int(plm_layer)}"
    weights_filename = "ae_normalized.pt" if normalized else "ae_unnormalized.pt"
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{layer_dir}/config.json",
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{layer_dir}/{weights_filename}",
        cache_dir=None if cache_dir is None else str(cache_dir),
    )

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    architecture = config["architecture"]
    model = InterPLMSparseAutoencoder(
        input_dim=int(architecture["esm_dim"]),
        feature_dim=int(architecture["feature_dim"]),
    )
    state_dict = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
