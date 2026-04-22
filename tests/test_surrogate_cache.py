from types import SimpleNamespace
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sqlalchemy")
pytest.importorskip("transformers")
import transformers
import torch.nn as nn

from prospero import surrogate
from prospero.probe_benchmark.interplm import InterPLMSparseAutoencoder
from prospero.runners import run_protein, run_variable_k


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

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        del attention_mask
        self.forward_call_count += 1
        offsets = torch.arange(self.config.hidden_size, device=input_ids.device).view(
            1, 1, -1
        )
        base_hidden_state = input_ids.unsqueeze(-1).float() + offsets
        hidden_states = tuple(
            base_hidden_state + float(layer_index) for layer_index in range(7)
        )
        return SimpleNamespace(
            last_hidden_state=hidden_states[-1],
            hidden_states=hidden_states if output_hidden_states else None,
        )


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
    model = surrogate.FrozenESMMeanPooledModel(
        make_args(**arg_overrides),
        tokenizer=FakeTokenizer(),
        esm=fake_esm,
    )
    return model, fake_esm


def make_fake_sae(input_dim, feature_dim=5):
    sae = InterPLMSparseAutoencoder(input_dim=input_dim, feature_dim=feature_dim)
    with torch.no_grad():
        sae.bias.zero_()
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
        for feature_idx in range(min(input_dim, feature_dim)):
            sae.encoder.weight[feature_idx, feature_idx] = 1.0
        sae.decoder.weight.zero_()
    sae.eval()
    return sae


def build_per_residue_model(monkeypatch, tmp_path, seq_length=5, **arg_overrides):
    fake_esm = FakeESM()
    monkeypatch.setattr(
        surrogate, "get_esm_embedding_cache_path", lambda: tmp_path / "esm_embeddings"
    )
    model = surrogate.FrozenESMPerResidueCNNModel(
        seq_length,
        make_args(**arg_overrides),
        tokenizer=FakeTokenizer(),
        esm=fake_esm,
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
    assert torch.allclose(first_embeddings, second_embeddings, atol=1e-2, rtol=1e-3)
    assert torch.allclose(first_embeddings[0], first_embeddings[2], atol=1e-2, rtol=1e-3)
    assert predictions.shape == (2,)
    assert not list((tmp_path / "esm_embeddings").rglob("*.pt"))


def test_training_uses_cached_embeddings_across_retrains(monkeypatch, tmp_path):
    model, fake_esm = build_model(monkeypatch, tmp_path)

    dataset = SimpleNamespace(
        train=np.array(["ACD", "AAA", "ACD"], dtype=object),
        train_scores=np.array([1.0, 2.0, 1.5], dtype=np.float32),
        valid=np.array(["CCC"], dtype=object),
        valid_scores=np.array([0.5], dtype=np.float32),
    )

    model.train(dataset)
    first_train_calls = fake_esm.forward_call_count
    assert 2 <= first_train_calls <= 3

    model.train(dataset)
    assert fake_esm.forward_call_count == first_train_calls


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
    assert not list((tmp_path / "esm_embeddings").rglob("*.pt"))


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
    first_train_calls = fake_esm.forward_call_count
    assert 2 <= first_train_calls <= 3

    model.train(dataset)
    assert fake_esm.forward_call_count == first_train_calls


def test_build_surrogate_model_supports_frozen_esm_cnn(monkeypatch, tmp_path):
    fake_esm = FakeESM(hidden_size=10)
    monkeypatch.setattr(
        surrogate, "get_esm_embedding_cache_path", lambda: tmp_path / "esm_embeddings"
    )

    model = surrogate.build_surrogate_model(
        5,
        make_args(surrogate_arch="frozen_esm_cnn"),
        shared_esm_components=surrogate.SharedESMComponents(
            tokenizer=FakeTokenizer(),
            esm=fake_esm,
            esm_forward_lock=threading.Lock(),
        ),
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
        shared_esm_components=surrogate.SharedESMComponents(
            tokenizer=FakeTokenizer(),
            esm=fake_esm,
            esm_forward_lock=threading.Lock(),
        ),
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
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name: fake_esm,
    )

    args = make_args(
        surrogate_arch=surrogate_arch,
        proxy_batch_size=2,
        disable_esm_cache=True,
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
    assert not list((tmp_path / "esm_embeddings").rglob("*.pt"))


def test_eval_path_does_not_write_cache(monkeypatch, tmp_path):
    model, fake_esm = build_model(monkeypatch, tmp_path)
    predictions = model.get_fitness(["AAA", "CCC"])
    assert predictions.shape == (2,)
    assert fake_esm.forward_call_count == 1
    assert not list((tmp_path / "esm_embeddings").rglob("*.pt"))


def test_training_splits_persistent_and_run_scoped_cache(monkeypatch, tmp_path):
    model, fake_esm = build_model(monkeypatch, tmp_path)
    model.cache_allowed_sequences = {"AAA"}

    _ = model.get_pooled_sequence_embeddings(["AAA", "BBB"])
    assert fake_esm.forward_call_count == 2
    _ = model.get_pooled_sequence_embeddings(["AAA", "BBB"])

    loaded, missing = model.persistent_embedding_cache.get_many(["AAA", "BBB"])
    assert "AAA" in loaded
    assert missing == ["BBB"]
    assert fake_esm.forward_call_count == 3


def test_build_surrogate_model_supports_interplm_mean_pool_ridge(
    monkeypatch, tmp_path
):
    fake_esm = FakeESM(hidden_size=6)
    monkeypatch.setattr(
        surrogate, "get_esm_embedding_cache_path", lambda: tmp_path / "esm_embeddings"
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name: fake_esm,
    )
    monkeypatch.setattr(
        surrogate,
        "load_interplm_sae",
        lambda **kwargs: make_fake_sae(input_dim=fake_esm.config.hidden_size),
    )

    model = surrogate.build_surrogate_model(
        5,
        make_args(
            surrogate_arch="interplm_mean_pool_ridge",
            interplm_layer=2,
            interplm_repo_id="fake/interplm",
            interplm_normalized=True,
            sae_token_chunk_size=3,
        ),
    )

    assert isinstance(model, surrogate.InterPLMMeanPooledSklearnModel)
    assert model.interplm_layer == 2
    assert model.dataset_representation_store is None


def test_interplm_ridge_surrogate_fit_predict(monkeypatch, tmp_path):
    fake_esm = FakeESM(hidden_size=6)
    monkeypatch.setattr(
        surrogate, "get_esm_embedding_cache_path", lambda: tmp_path / "esm_embeddings"
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name: fake_esm,
    )
    monkeypatch.setattr(
        surrogate,
        "load_interplm_sae",
        lambda **kwargs: make_fake_sae(input_dim=fake_esm.config.hidden_size),
    )

    model = surrogate.build_surrogate_model(
        5,
        make_args(
            surrogate_arch="interplm_mean_pool_ridge",
            interplm_layer=2,
            interplm_repo_id="fake/interplm",
            interplm_normalized=False,
            sae_token_chunk_size=4,
            disable_esm_cache=True,
        ),
    )
    dataset = SimpleNamespace(
        train=np.array(["ACDEF", "AAAAC", "CCCCC"], dtype=object),
        train_scores=np.array([1.0, 2.0, 0.5], dtype=np.float32),
        valid=np.array(["DDDDE"], dtype=object),
        valid_scores=np.array([0.2], dtype=np.float32),
    )

    model.train(dataset)
    pooled = model.get_sequence_representations(["ACDEF", "DDDDE"])
    predictions = model.get_fitness(["ACDEF", "DDDDE"])

    assert pooled.shape == (2, 5)
    assert predictions.shape == (2,)
    assert torch.isfinite(predictions).all()


def test_runner_plumbing_preserves_interplm_configuration(tmp_path):
    runner_args = SimpleNamespace(
        task="LGK",
        alphabet="RANDOM",
        surrogate_arch="interplm_mean_pool_ridge",
        min_corruptions=3,
        max_corruptions=10,
        esm_cnn_projection_dim=None,
        esm_cnn_use_layernorm=False,
        esm_cnn_concat_one_hot=False,
        esm_model_name="facebook/esm2_t6_8M_UR50D",
        esm_max_length=None,
        ridge_alpha=0.01,
        ridge_fit_intercept=True,
        interplm_layer=2,
        interplm_repo_id="Elana/InterPLM-esm2-8m",
        interplm_normalized=True,
        sae_token_chunk_size=2048,
        disable_esm_cache=False,
        debug_events=False,
        debug_heartbeat_seconds=15.0,
    )

    protein_args = run_variable_k._build_protein_args(
        runner_args=runner_args,
        batch_dir=tmp_path / "results",
        n_iters=10,
        n_queries=8,
        seed=5,
    )

    assert protein_args.surrogate_arch == "interplm_mean_pool_ridge"
    assert protein_args.interplm_layer == 2
    assert protein_args.interplm_repo_id == "Elana/InterPLM-esm2-8m"
    assert protein_args.interplm_normalized is True
    assert protein_args.sae_token_chunk_size == 2048
    parsed = run_protein.get_parser().parse_args(
        ["--surrogate_arch", "interplm_mean_pool_ridge", "--interplm_layer", "2"]
    )
    assert parsed.surrogate_arch == "interplm_mean_pool_ridge"
    assert parsed.interplm_layer == 2


def test_flat_esm_dataset_store_reuses_embeddings_across_model_instances(
    monkeypatch, tmp_path
):
    ordered_sequences = ["ACDEF", "AAAAC", "CCCCC", "DDDDE"]
    dataset = SimpleNamespace(
        train=np.array(["ACDEF", "AAAAC", "CCCCC"], dtype=object),
        train_scores=np.array([1.0, 2.0, 0.5], dtype=np.float32),
        valid=np.array(["DDDDE"], dtype=object),
        valid_scores=np.array([0.2], dtype=np.float32),
    )

    monkeypatch.setattr(
        surrogate, "get_esm_embedding_cache_path", lambda: tmp_path / "esm_embeddings"
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda model_name, **kwargs: FakeTokenizer(),
    )

    first_fake_esm = FakeESM(hidden_size=6)
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name, **kwargs: first_fake_esm,
    )
    first_model = surrogate.build_surrogate_model(
        5,
        make_args(
            surrogate_arch="frozen_esm_flat_ridge",
            proxy_batch_size=2,
            cache_allowed_sequences=set(ordered_sequences),
            cache_allowed_sequences_ordered=ordered_sequences,
            dataset_cache_task="LGK",
        ),
    )
    first_model.train(dataset)
    assert first_fake_esm.forward_call_count > 0
    assert list((tmp_path / "dataset_representations").rglob("representations.npy"))

    second_fake_esm = FakeESM(hidden_size=6)
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name, **kwargs: second_fake_esm,
    )
    second_model = surrogate.build_surrogate_model(
        5,
        make_args(
            surrogate_arch="frozen_esm_flat_ridge",
            proxy_batch_size=2,
            cache_allowed_sequences=set(ordered_sequences),
            cache_allowed_sequences_ordered=ordered_sequences,
            dataset_cache_task="LGK",
        ),
    )
    second_model.train(dataset)

    assert second_fake_esm.forward_call_count == 0


def test_interplm_dataset_store_reuses_embeddings_across_model_instances(
    monkeypatch, tmp_path
):
    ordered_sequences = ["ACDEF", "AAAAC", "CCCCC", "DDDDE"]
    dataset = SimpleNamespace(
        train=np.array(["ACDEF", "AAAAC", "CCCCC"], dtype=object),
        train_scores=np.array([1.0, 2.0, 0.5], dtype=np.float32),
        valid=np.array(["DDDDE"], dtype=object),
        valid_scores=np.array([0.2], dtype=np.float32),
    )

    monkeypatch.setattr(
        surrogate, "get_esm_embedding_cache_path", lambda: tmp_path / "esm_embeddings"
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda model_name, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        surrogate,
        "load_interplm_sae",
        lambda **kwargs: make_fake_sae(input_dim=6),
    )

    first_fake_esm = FakeESM(hidden_size=6)
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name, **kwargs: first_fake_esm,
    )
    first_model = surrogate.build_surrogate_model(
        5,
        make_args(
            surrogate_arch="interplm_mean_pool_ridge",
            interplm_layer=2,
            interplm_repo_id="fake/interplm",
            interplm_normalized=True,
            sae_token_chunk_size=4,
            proxy_batch_size=2,
            cache_allowed_sequences=set(ordered_sequences),
            cache_allowed_sequences_ordered=ordered_sequences,
            dataset_cache_task="LGK",
        ),
    )
    first_model.train(dataset)
    assert first_fake_esm.forward_call_count > 0
    assert list((tmp_path / "dataset_representations").rglob("representations.npy"))

    second_fake_esm = FakeESM(hidden_size=6)
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name, **kwargs: second_fake_esm,
    )
    second_model = surrogate.build_surrogate_model(
        5,
        make_args(
            surrogate_arch="interplm_mean_pool_ridge",
            interplm_layer=2,
            interplm_repo_id="fake/interplm",
            interplm_normalized=True,
            sae_token_chunk_size=4,
            proxy_batch_size=2,
            cache_allowed_sequences=set(ordered_sequences),
            cache_allowed_sequences_ordered=ordered_sequences,
            dataset_cache_task="LGK",
        ),
    )
    second_model.train(dataset)

    assert second_fake_esm.forward_call_count == 0
