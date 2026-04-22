from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from jaxtyping import Float, Int64, jaxtyped
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerBase

Tensor = torch.Tensor


@dataclass(frozen=True)
class LatentDirectionSearchResult:
    hidden_states: Float[Tensor, "num_scales batch seq hidden"]
    token_ids: Int64[Tensor, "num_scales batch seq"] | None
    sequences: list[list[str]] | None


@dataclass(frozen=True)
class MaskedLatentDirectionSearchResult:
    token_ids: Int64[Tensor, "num_directions num_scales batch seq"]
    pred_token_log_probs: Float[Tensor, "num_directions num_scales batch seq"]


class EsmPrefix(nn.Module):
    def __init__(self, layers: nn.ModuleList, n: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers[: n + 1])

    def forward(
        self,
        hidden: Float[Tensor, "batch seq hidden"],
        attention_mask: Tensor | None,
        encoder_attention_mask: Tensor | None,
    ) -> Float[Tensor, "batch seq hidden"]:
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=encoder_attention_mask,
            )
        return hidden


class EsmSuffix(nn.Module):
    def __init__(
        self,
        layers: nn.ModuleList,
        n: int,
        final_norm: nn.Module | None,
        lm_head: nn.Module,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers[n + 1 :])
        self.final_norm = final_norm
        self.lm_head = lm_head

    def forward(
        self,
        hidden: Float[Tensor, "scaled_batch seq hidden"],
        attention_mask: Tensor | None,
        encoder_attention_mask: Tensor | None,
    ) -> tuple[
        Float[Tensor, "scaled_batch seq hidden"],
        Float[Tensor, "scaled_batch seq vocab"],
    ]:
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=encoder_attention_mask,
            )
        if self.final_norm is not None:
            hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        return hidden, logits


_COMPILED_SPLIT: dict[tuple[int, int], tuple[nn.Module, nn.Module]] = {}
_EAGER_SPLIT: dict[tuple[int, int], tuple[nn.Module, nn.Module]] = {}


def _get_compiled_split(model: nn.Module, n: int) -> tuple[nn.Module, nn.Module]:
    key = (id(model), n)
    cached = _COMPILED_SPLIT.get(key)
    if cached is not None:
        return cached

    layers = model.esm.encoder.layer
    prefix = EsmPrefix(layers, n).eval()
    suffix = EsmSuffix(
        layers=layers,
        n=n,
        final_norm=model.esm.encoder.emb_layer_norm_after,
        lm_head=model.lm_head,
    ).eval()

    compiled_prefix = cast(nn.Module, torch.compile(prefix, mode="reduce-overhead"))
    compiled_suffix = cast(nn.Module, torch.compile(suffix, mode="reduce-overhead"))
    _COMPILED_SPLIT[key] = (compiled_prefix, compiled_suffix)
    return compiled_prefix, compiled_suffix


def _get_eager_split(model: nn.Module, n: int) -> tuple[nn.Module, nn.Module]:
    key = (id(model), n)
    cached = _EAGER_SPLIT.get(key)
    if cached is not None:
        return cached

    layers = model.esm.encoder.layer
    prefix = EsmPrefix(layers, n).eval()
    suffix = EsmSuffix(
        layers=layers,
        n=n,
        final_norm=model.esm.encoder.emb_layer_norm_after,
        lm_head=model.lm_head,
    ).eval()
    _EAGER_SPLIT[key] = (prefix, suffix)
    return prefix, suffix


@jaxtyped(typechecker=None)
@torch.no_grad()
def latent_direction_search(
    model: nn.Module,
    n: int,
    steering_vector: Float[Tensor, "hidden"],
    s_values: Float[Tensor, "num_scales"],
    inputs_embeds: Float[Tensor, "batch seq hidden"],
    attention_mask: Int64[Tensor, "batch seq"] | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> LatentDirectionSearchResult:
    """Run prefix once, steer, then run suffix batched over scalars."""
    device = next(model.parameters()).device
    model = model.eval()

    layers = model.esm.encoder.layer
    num_layers = len(layers)
    if n < 0 or n >= num_layers:
        raise ValueError(f"n must be in [0, {num_layers - 1}], got {n}")
    batch_size, seq_len, _ = inputs_embeds.shape

    num_scales = int(s_values.shape[0])

    inputs_embeds = inputs_embeds.to(device)
    attention_mask = attention_mask.to(device) if attention_mask is not None else None
    steering_vector = steering_vector.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    s_values = s_values.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    attn_mask, enc_attn_mask = model.esm._create_attention_masks(
        attention_mask=attention_mask,
        encoder_attention_mask=None,
        embedding_output=inputs_embeds,
        encoder_hidden_states=None,
        cache_position=torch.arange(seq_len, device=device),
        past_key_values=None,
    )

    prefix, suffix = _get_compiled_split(model, n)
    torch.compiler.cudagraph_mark_step_begin()
    hidden = prefix(inputs_embeds, attn_mask, enc_attn_mask).clone()

    steered = hidden.unsqueeze(0) + s_values[:, None, None, None] * steering_vector[None, None, None, :]
    steered = steered.reshape(num_scales * batch_size, seq_len, -1)

    if attn_mask is not None:
        attn_mask = attn_mask.repeat(num_scales, *(1 for _ in range(attn_mask.ndim - 1)))
    if enc_attn_mask is not None:
        enc_attn_mask = enc_attn_mask.repeat(num_scales, *(1 for _ in range(enc_attn_mask.ndim - 1)))
    torch.compiler.cudagraph_mark_step_begin()
    steered, logits = suffix(steered, attn_mask, enc_attn_mask)

    hidden = steered.reshape(num_scales, batch_size, seq_len, -1)

    token_ids: Int64[Tensor, "num_scales batch seq"] | None = None
    sequences: list[list[str]] | None = None

    logits = logits.reshape(num_scales, batch_size, seq_len, -1)
    token_ids = torch.argmax(logits, dim=-1)
    if tokenizer is not None:
        sequences = [
            tokenizer.batch_decode(token_ids[i], skip_special_tokens=True)
            for i in range(num_scales)
        ]

    return LatentDirectionSearchResult(hidden_states=hidden, token_ids=token_ids, sequences=sequences)


@torch._dynamo.disable
@jaxtyped(typechecker=None)
@torch.no_grad()
def masked_latent_direction_search(
    model: nn.Module,
    n: int,
    steering_directions: Float[Tensor, "num_directions hidden"],
    s_values: Float[Tensor, "num_scales"],
    inputs_embeds: Float[Tensor, "batch seq hidden"],
    token_steering_mask: Int64[Tensor, "batch seq"] | torch.BoolTensor,
    attention_mask: Int64[Tensor, "batch seq"] | None = None,
    combo_chunk_size: int = 4,
) -> MaskedLatentDirectionSearchResult:
    """Prefix once, then steer only masked tokens for many (direction, scalar) combos."""
    device = next(model.parameters()).device
    model = model.eval()

    layers = model.esm.encoder.layer
    num_layers = len(layers)
    if n < 0 or n >= num_layers:
        raise ValueError(f"n must be in [0, {num_layers - 1}], got {n}")

    batch_size, seq_len, _ = inputs_embeds.shape
    num_directions = int(steering_directions.shape[0])
    num_scales = int(s_values.shape[0])
    num_combos = num_directions * num_scales

    inputs_embeds = inputs_embeds.to(device)
    attention_mask = attention_mask.to(device) if attention_mask is not None else None
    token_steering_mask = token_steering_mask.to(device).bool()
    steering_directions = steering_directions.to(
        device=inputs_embeds.device, dtype=inputs_embeds.dtype
    )
    s_values = s_values.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    attn_mask, enc_attn_mask = model.esm._create_attention_masks(
        attention_mask=attention_mask,
        encoder_attention_mask=None,
        embedding_output=inputs_embeds,
        encoder_hidden_states=None,
        cache_position=torch.arange(seq_len, device=device),
        past_key_values=None,
    )
    hidden = inputs_embeds
    for i in range(n + 1):
        hidden = layers[i](
            hidden,
            attention_mask=attn_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=enc_attn_mask,
        )
    hidden = hidden.clone()

    combo_vectors = (
        steering_directions[:, None, :] * s_values[None, :, None]
    ).reshape(num_combos, -1)
    token_mask = token_steering_mask.to(dtype=hidden.dtype)

    token_ids_flat = torch.empty(
        (num_combos, batch_size, seq_len), device=device, dtype=torch.long
    )
    pred_log_probs_flat = torch.empty(
        (num_combos, batch_size, seq_len), device=device, dtype=hidden.dtype
    )

    step = max(1, int(combo_chunk_size))
    for start in range(0, num_combos, step):
        stop = min(num_combos, start + step)
        vectors = combo_vectors[start:stop]
        count = stop - start

        steered = hidden.unsqueeze(0) + (
            vectors[:, None, None, :] * token_mask[None, :, :, None]
        )
        steered = steered.reshape(count * batch_size, seq_len, -1)

        attn_mask_chunk = (
            attn_mask.repeat(count, *(1 for _ in range(attn_mask.ndim - 1)))
            if attn_mask is not None
            else None
        )
        enc_attn_mask_chunk = (
            enc_attn_mask.repeat(count, *(1 for _ in range(enc_attn_mask.ndim - 1)))
            if enc_attn_mask is not None
            else None
        )

        for i in range(n + 1, num_layers):
            steered = layers[i](
                steered,
                attention_mask=attn_mask_chunk,
                encoder_hidden_states=None,
                encoder_attention_mask=enc_attn_mask_chunk,
            )
        if model.esm.encoder.emb_layer_norm_after is not None:
            steered = model.esm.encoder.emb_layer_norm_after(steered)
        logits_chunk = model.lm_head(steered)
        token_chunk = torch.argmax(logits_chunk, dim=-1)
        log_probs_chunk = torch.log_softmax(logits_chunk, dim=-1)
        pred_logp_chunk = torch.gather(
            log_probs_chunk, dim=-1, index=token_chunk.unsqueeze(-1)
        ).squeeze(-1)

        token_ids_flat[start:stop] = token_chunk.reshape(count, batch_size, seq_len)
        pred_log_probs_flat[start:stop] = pred_logp_chunk.reshape(
            count, batch_size, seq_len
        )

    return MaskedLatentDirectionSearchResult(
        token_ids=token_ids_flat.reshape(num_directions, num_scales, batch_size, seq_len),
        pred_token_log_probs=pred_log_probs_flat.reshape(
            num_directions, num_scales, batch_size, seq_len
        ),
    )


@torch.no_grad()
def main() -> None:
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    seqs = ["MKTAYIAKQRQISFVKSHFSRQ"]
    inputs = tokenizer(seqs, return_tensors="pt").to(device)
    inputs_embeds = model.esm.embeddings(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )

    n = 3
    hidden_size = int(model.config.hidden_size)
    steering = torch.randn(hidden_size, device=device)
    s_values = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device=device)

    result = latent_direction_search(
        model=model,
        n=n,
        steering_vector=steering,
        s_values=s_values,
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.get("attention_mask"),
        tokenizer=tokenizer,
    )

    print("hidden shape:", tuple(result.hidden_states.shape))
    if result.sequences is not None:
        for i, s in enumerate(s_values.tolist()):
            print(f"s={s:>5.1f} -> {result.sequences[i][0]}")


if __name__ == "__main__":
    main()
