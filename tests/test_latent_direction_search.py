from __future__ import annotations

import torch
from torch import nn

from latent_search.latent_direction_search import masked_latent_direction_search


class _IdentityLayer(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = attention_mask, encoder_hidden_states, encoder_attention_mask
        return hidden


class _FakeEncoder(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList([_IdentityLayer() for _ in range(num_layers)])
        self.emb_layer_norm_after: nn.Module | None = None


class _FakeEsm(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.encoder = _FakeEncoder(num_layers=num_layers)

    def _create_attention_masks(
        self,
        attention_mask: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        embedding_output: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        cache_position: torch.Tensor,
        past_key_values: object | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        _ = (
            attention_mask,
            encoder_attention_mask,
            embedding_output,
            encoder_hidden_states,
            cache_position,
            past_key_values,
        )
        return None, None


class _FakeModel(nn.Module):
    def __init__(self, hidden_size: int = 2, vocab_size: int = 2, num_layers: int = 3) -> None:
        super().__init__()
        self.esm = _FakeEsm(num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=True)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.bias.zero_()
            # logit(token_0) = 0
            # logit(token_1) = hidden_dim_0
            self.lm_head.weight[1, 0] = 1.0


def test_masked_latent_direction_search_steers_only_masked_positions() -> None:
    model = _FakeModel()
    inputs_embeds = torch.zeros((1, 3, 2), dtype=torch.float32)
    attention_mask = torch.ones((1, 3), dtype=torch.int64)
    token_steering_mask = torch.tensor([[0, 1, 0]], dtype=torch.int64)

    steering_directions = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    s_values = torch.tensor([0.0, 3.0], dtype=torch.float32)
    result = masked_latent_direction_search(
        model=model,
        n=1,
        steering_directions=steering_directions,
        s_values=s_values,
        inputs_embeds=inputs_embeds,
        token_steering_mask=token_steering_mask,
        attention_mask=attention_mask,
        combo_chunk_size=1,
    )

    # Shapes: [num_directions, num_scales, batch, seq]
    assert tuple(result.token_ids.shape) == (1, 2, 1, 3)
    assert tuple(result.pred_token_log_probs.shape) == (1, 2, 1, 3)

    token_ids = result.token_ids[0]  # [num_scales, batch, seq]

    # Scalar 0.0 -> no steering, all tokens choose token_0.
    assert token_ids[0, 0, :].tolist() == [0, 0, 0]

    # Scalar 3.0 -> only the masked middle token flips to token_1.
    assert token_ids[1, 0, :].tolist() == [0, 1, 0]

