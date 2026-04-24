from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def is_evolutionaryscale_model(model_id: str) -> bool:
    model_id = str(model_id)
    return model_id.startswith("EvolutionaryScale/esm3") or model_id.startswith(
        "EvolutionaryScale/esmc"
    )


def hf_to_esm_sdk_name(model_id: str) -> str:
    if model_id == "EvolutionaryScale/esm3-sm-open-v1":
        return "esm3_sm_open_v1"
    if model_id == "EvolutionaryScale/esmc-300m-2024-12":
        return "esmc_300m"
    if model_id == "EvolutionaryScale/esmc-600m-2024-12":
        return "esmc_600m"
    raise ValueError(
        f"Unsupported EvolutionaryScale model mapping for '{model_id}'. "
        "Supported currently: esm3-sm-open-v1, esmc-300m-2024-12, esmc-600m-2024-12."
    )


@dataclass(frozen=True)
class EvolutionaryScaleBackend:
    model_id: str
    sdk_name: str
    device: str
    model: object
    hidden_size: int

    @staticmethod
    def load(model_id: str, *, device: str) -> "EvolutionaryScaleBackend":
        try:
            from esm.models.esm3 import ESM3
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein
            from esm.sdk.api import LogitsConfig
        except Exception as exc:
            raise ImportError(
                "EvolutionaryScale backend requires esm>=3.x. "
                "Run with `uv run --with \"esm==3.2.3\" ...` or add it to the environment."
            ) from exc

        sdk_name = hf_to_esm_sdk_name(model_id)
        torch_device = torch.device(device)
        if sdk_name.startswith("esmc_"):
            model = ESMC.from_pretrained(sdk_name, device=torch_device)
        else:
            model = ESM3.from_pretrained(sdk_name, device=torch_device)

        # Infer hidden size once.
        probe = ESMProtein(sequence="ACDEFGHIKLMNPQRSTVWY")
        probe_tensor = model.encode(probe)
        probe_out = model.logits(
            probe_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
        probe_embeddings = probe_out.embeddings
        hidden_size = int(probe_embeddings.shape[-1])

        return EvolutionaryScaleBackend(
            model_id=model_id,
            sdk_name=sdk_name,
            device=device,
            model=model,
            hidden_size=hidden_size,
        )

    def _residue_embeddings_for_sequence(self, sequence: str) -> torch.Tensor:
        from esm.sdk.api import ESMProtein
        from esm.sdk.api import LogitsConfig

        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        output = self.model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
        embeddings = output.embeddings
        if embeddings.dim() != 3 or embeddings.shape[0] != 1:
            raise ValueError(
                f"Unexpected embedding tensor shape from EvolutionaryScale backend: {tuple(embeddings.shape)}"
            )
        residue_embeddings = embeddings[0]

        seq_len = len(sequence)
        if residue_embeddings.shape[0] == seq_len + 2:
            residue_embeddings = residue_embeddings[1:-1]
        elif residue_embeddings.shape[0] == seq_len:
            pass
        else:
            raise ValueError(
                f"Unexpected token length {residue_embeddings.shape[0]} for sequence length {seq_len}."
            )
        return residue_embeddings.to(torch.float32)

    @torch.no_grad()
    def compute_representations(
        self,
        sequences: Sequence[str],
        *,
        representation_name: str,
        expected_sequence_length: int | None = None,
    ) -> torch.Tensor:
        if not sequences:
            if representation_name == "mean_pool_residue_embeddings_v1":
                return torch.empty((0, self.hidden_size), dtype=torch.float32)
            seq_len = int(expected_sequence_length or 0)
            return torch.empty((0, seq_len, self.hidden_size), dtype=torch.float32)

        residue_batch = [self._residue_embeddings_for_sequence(sequence) for sequence in sequences]

        if expected_sequence_length is not None:
            for emb in residue_batch:
                if emb.shape[0] != expected_sequence_length:
                    raise ValueError(
                        "Per-residue embeddings length mismatch. "
                        f"Expected {expected_sequence_length}, got {emb.shape[0]}."
                    )

        if representation_name == "mean_pool_residue_embeddings_v1":
            return torch.stack([emb.mean(dim=0) for emb in residue_batch], dim=0)
        if representation_name == "per_residue_embeddings_v1":
            reference = residue_batch[0].shape[0]
            if any(emb.shape[0] != reference for emb in residue_batch):
                raise ValueError(
                    "Per-residue backend expects fixed sequence length within each batch."
                )
            return torch.stack(residue_batch, dim=0)
        raise ValueError(f"Unsupported representation_name={representation_name}")
