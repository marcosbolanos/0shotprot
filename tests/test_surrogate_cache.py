from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sqlalchemy")
pytest.importorskip("transformers")
import torch.nn as nn

from prospero import surrogate


class FakeTokenizer:
    def __call__(
        self,
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=None,
    ):
        token_ids_per_sequence = []
        for sequence in sequences:
            residue_ids = [ord(token) % 20 + 3 for token in sequence]
            if truncation and max_length is not None:
                residue_ids = residue_ids[: max(max_length - 2, 0)]
            token_ids_per_sequence.append([1, *residue_ids, 2])

        max_token_count = max(len(token_ids) for token_ids in token_ids_per_sequence)
        padded_token_ids = []
        attention_masks = []
        for token_ids in token_ids_per_sequence:
            pad_count = max_token_count - len(token_ids)
            padded_token_ids.append(token_ids + [0] * pad_count)
            attention_masks.append([1] * len(token_ids) + [0] * pad_count)

        return {
            "input_ids": torch.tensor(padded_token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


class FakeESM(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.forward_call_count = 0

    def forward(self, input_ids, attention_mask):
        del attention_mask
        self.forward_call_count += 1
        offsets = torch.arange(self.config.hidden_size, device=input_ids.device).view(
            1, 1, -1
        )
        last_hidden_state = input_ids.unsqueeze(-1).float() + offsets
        return SimpleNamespace(last_hidden_state=last_hidden_state)


def make_args(**overrides):
    args = {
        "esm_model_name": "fake-esm",
        "esm_max_length": None,
        "esm_mlp_hidden_dim": 4,
        "esm_mlp_dropout": 0.0,
        "esm_cnn_projection_dim": None,
        "esm_cnn_use_layernorm": False,
        "esm_cnn_concat_one_hot": False,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "proxy_batch_size": 2,
        "num_model_max_epochs": 3,
        "epochs_per_valid": 1,
        "patience": 5,
        "ridge_alpha": 1.0,
        "ridge_fit_intercept": True,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def build_model(monkeypatch, tmp_path, **arg_overrides):
    fake_esm = FakeESM()
    monkeypatch.setattr(
        surrogate, "get_esm_embedding_cache_path", lambda: tmp_path / "esm_embeddings"
    )
    monkeypatch.setattr(
        surrogate.AutoTokenizer,
        "from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        surrogate.AutoModel,
        "from_pretrained",
        lambda model_name: fake_esm,
    )
    model = surrogate.FrozenESMMeanPooledModel(make_args(**arg_overrides))
    return model, fake_esm


def build_per_residue_model(monkeypatch, tmp_path, seq_length=5, **arg_overrides):
    fake_esm = FakeESM()
    monkeypatch.setattr(
        surrogate, "get_esm_embedding_cache_path", lambda: tmp_path / "esm_embeddings"
    )
    monkeypatch.setattr(
        surrogate.AutoTokenizer,
        "from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        surrogate.AutoModel,
        "from_pretrained",
        lambda model_name: fake_esm,
    )
    model = surrogate.FrozenESMPerResidueCNNModel(
        seq_length, make_args(**arg_overrides)
    )
    return model, fake_esm


def test_cached_embeddings_are_reused(monkeypatch, tmp_path):
    model, fake_esm = build_model(monkeypatch, tmp_path)

    sequences = ["ACD", "AAA", "ACD"]
    first_embeddings = model.get_pooled_sequence_embeddings(sequences)
    assert fake_esm.forward_call_count == 1

    second_embeddings = model.get_pooled_sequence_embeddings(sequences)
    assert fake_esm.forward_call_count == 1

    predictions = model.get_fitness(["AAA", "CCC"])
    assert fake_esm.forward_call_count == 2

    assert first_embeddings.shape == (3, fake_esm.config.hidden_size)
    assert torch.allclose(first_embeddings, second_embeddings)
    assert torch.allclose(first_embeddings[0], first_embeddings[2])
    assert predictions.shape == (2,)
    assert list((tmp_path / "esm_embeddings").rglob("*.pt"))


def test_training_uses_cached_embeddings_across_retrains(monkeypatch, tmp_path):
    model, fake_esm = build_model(monkeypatch, tmp_path)

    dataset = SimpleNamespace(
        train=np.array(["ACD", "AAA", "ACD"], dtype=object),
        train_scores=np.array([1.0, 2.0, 1.5], dtype=np.float32),
        valid=np.array(["CCC"], dtype=object),
        valid_scores=np.array([0.5], dtype=np.float32),
    )

    model.train(dataset)
    assert fake_esm.forward_call_count == 2

    model.train(dataset)
    assert fake_esm.forward_call_count == 2


def test_cached_per_residue_embeddings_are_reused(monkeypatch, tmp_path):
    model, fake_esm = build_per_residue_model(monkeypatch, tmp_path, seq_length=5)

    sequences = ["ACDEF", "AAAAC", "ACDEF"]
    first_embeddings = model.get_residue_sequence_embeddings(sequences)
    assert fake_esm.forward_call_count == 1

    second_embeddings = model.get_residue_sequence_embeddings(sequences)
    assert fake_esm.forward_call_count == 1

    predictions = model.get_fitness(["AAAAC", "CCCCC"])
    assert fake_esm.forward_call_count == 2

    assert first_embeddings.shape == (3, 5, fake_esm.config.hidden_size)
    assert torch.allclose(first_embeddings, second_embeddings)
    assert torch.allclose(first_embeddings[0], first_embeddings[2])
    assert predictions.shape == (2,)
    assert list((tmp_path / "esm_embeddings").rglob("*.pt"))


def test_per_residue_training_uses_cached_embeddings_across_retrains(
    monkeypatch, tmp_path
):
    model, fake_esm = build_per_residue_model(monkeypatch, tmp_path, seq_length=5)

    dataset = SimpleNamespace(
        train=np.array(["ACDEF", "AAAAC", "ACDEF"], dtype=object),
        train_scores=np.array([1.0, 2.0, 1.5], dtype=np.float32),
        valid=np.array(["CCCCC"], dtype=object),
        valid_scores=np.array([0.5], dtype=np.float32),
    )

    model.train(dataset)
    assert fake_esm.forward_call_count == 2

    model.train(dataset)
    assert fake_esm.forward_call_count == 2


def test_build_surrogate_model_supports_frozen_esm_cnn(monkeypatch, tmp_path):
    fake_esm = FakeESM(hidden_size=10)
    monkeypatch.setattr(
        surrogate, "get_esm_embedding_cache_path", lambda: tmp_path / "esm_embeddings"
    )

    model = surrogate.build_surrogate_model(
        5,
        make_args(surrogate_arch="frozen_esm_cnn"),
        shared_esm_components=(FakeTokenizer(), fake_esm),
    )

    assert isinstance(model, surrogate.FrozenESMPerResidueCNNModel)
    assert model.net.conv_1.in_channels == fake_esm.config.hidden_size


def test_frozen_esm_cnn_projection_changes_input_channels(monkeypatch, tmp_path):
    fake_esm = FakeESM(hidden_size=10)
    monkeypatch.setattr(
        surrogate, "get_esm_embedding_cache_path", lambda: tmp_path / "esm_embeddings"
    )

    model = surrogate.build_surrogate_model(
        5,
        make_args(surrogate_arch="frozen_esm_cnn", esm_cnn_projection_dim=6),
        shared_esm_components=(FakeTokenizer(), fake_esm),
    )

    assert isinstance(model.input_adapter.projection, nn.Linear)
    assert model.net.conv_1.in_channels == 6


def test_frozen_esm_cnn_can_concatenate_one_hot_inputs(monkeypatch, tmp_path):
    model, fake_esm = build_per_residue_model(
        monkeypatch,
        tmp_path,
        seq_length=5,
        esm_cnn_projection_dim=6,
        esm_cnn_use_layernorm=True,
        esm_cnn_concat_one_hot=True,
    )

    predictions = model.get_fitness(["ACDEF", "AAAAC"])

    assert fake_esm.forward_call_count == 1
    assert model.net.conv_1.in_channels == 26
    assert predictions.shape == (2,)


@pytest.mark.parametrize(
    "surrogate_arch",
    ["frozen_esm_flat_linear", "frozen_esm_flat_ridge"],
)
def test_flattened_one_hot_sklearn_surrogate_fit_predict(
    monkeypatch, tmp_path, surrogate_arch
):
    fake_esm = FakeESM(hidden_size=6)
    monkeypatch.setattr(
        surrogate, "get_esm_embedding_cache_path", lambda: tmp_path / "esm_embeddings"
    )
    monkeypatch.setattr(
        surrogate.AutoTokenizer,
        "from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        surrogate.AutoModel,
        "from_pretrained",
        lambda model_name: fake_esm,
    )

    args = make_args(
        surrogate_arch=surrogate_arch,
        proxy_batch_size=2,
    )
    model = surrogate.build_surrogate_model(5, args)

    dataset = SimpleNamespace(
        train=np.array(["ACDEF", "AAAAC", "CCCCC"], dtype=object),
        train_scores=np.array([1.0, 2.0, 0.5], dtype=np.float32),
        valid=np.array(["DDDDE"], dtype=object),
        valid_scores=np.array([0.2], dtype=np.float32),
    )
    model.train(dataset)
    predictions = model.get_fitness(["ACDEF", "DDDDE"])

    assert predictions.shape == (2,)
    assert torch.isfinite(predictions).all()
    # Cache disabled by default unless an allowlist is provided.
    assert not (tmp_path / "esm_embeddings").exists()
