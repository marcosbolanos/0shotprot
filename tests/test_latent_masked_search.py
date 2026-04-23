from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from prospero.latent_masked_search import _extract_directions
from prospero.latent_masked_search import _build_rerank_scores
from prospero.latent_masked_search import _apply_mutation_cap_mask
from prospero.latent_masked_search import _apply_perplexity_cap_mask


class _FakeSae:
    def __init__(self) -> None:
        self.decoder = SimpleNamespace(weight=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


class _FakeModel:
    def __init__(self) -> None:
        self.regressor = SimpleNamespace(coef_=np.array([2.0, -3.0], dtype=np.float32))
        self.scaler = SimpleNamespace(scale_=np.array([1.0, 1.0], dtype=np.float32))
        self._sae = _FakeSae()

    def _get_sae(self) -> _FakeSae:
        return self._sae


class _FakeProxy:
    def __init__(self) -> None:
        self.models = [_FakeModel()]


def test_extract_directions_signed_mode_uses_coef_sign() -> None:
    proxy = _FakeProxy()
    directions, meta = _extract_directions(
        proxy=proxy,
        top_features=2,
        steering_direction_mode="signed",
    )
    assert tuple(directions.shape) == (2, 2)
    assert [m.feature_index for m in meta] == [1, 0]
    assert [m.direction_sign for m in meta] == [-1, 1]


def test_extract_directions_both_mode_expands_positive_and_negative() -> None:
    proxy = _FakeProxy()
    directions, meta = _extract_directions(
        proxy=proxy,
        top_features=2,
        steering_direction_mode="both",
    )
    assert tuple(directions.shape) == (4, 2)
    assert [m.feature_index for m in meta] == [1, 1, 0, 0]
    assert [m.direction_sign for m in meta] == [1, -1, 1, -1]


def test_build_rerank_scores_applies_mutation_penalty() -> None:
    surrogate = np.array([1.0, 1.0, 0.5], dtype=np.float64)
    mut = np.array([0.0, 3.0, 0.0], dtype=np.float64)
    scores = _build_rerank_scores(
        surrogate_scores=surrogate,
        mutation_counts=mut,
        mutation_penalty_lambda=0.2,
    )
    assert np.allclose(scores, np.array([1.0, 0.4, 0.5], dtype=np.float64))


def test_apply_mutation_cap_mask() -> None:
    mut = np.array([1.0, 3.0, 5.0], dtype=np.float64)
    assert _apply_mutation_cap_mask(mut, None).tolist() == [True, True, True]
    assert _apply_mutation_cap_mask(mut, 3).tolist() == [True, True, False]


def test_apply_perplexity_cap_mask() -> None:
    mean_ppl = np.array([3.0, 7.0, 5.0], dtype=np.float64)
    max_ppl = np.array([7.0, 9.0, 13.0], dtype=np.float64)
    assert _apply_perplexity_cap_mask(mean_ppl, max_ppl, None, None).tolist() == [True, True, True]
    assert _apply_perplexity_cap_mask(mean_ppl, max_ppl, 6.0, None).tolist() == [True, False, True]
    assert _apply_perplexity_cap_mask(mean_ppl, max_ppl, None, 10.0).tolist() == [True, True, False]
    assert _apply_perplexity_cap_mask(mean_ppl, max_ppl, 6.0, 10.0).tolist() == [True, False, False]
