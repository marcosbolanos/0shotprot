import numpy as np
import torch

from prospero.probe_benchmark.features import CachedTaskEmbeddingsLoader, build_feature_matrix
from prospero.probe_benchmark.metrics import regression_metrics, spearmanr
from prospero.probe_benchmark.models import ProbeSpec, fit_best_probe


def test_feature_builders_have_expected_shapes():
    residue_embeddings = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    mean_pool = build_feature_matrix("mean_pool", residue_embeddings)
    mean_max = build_feature_matrix("mean_max", residue_embeddings)
    flatten = build_feature_matrix("flatten", residue_embeddings)
    flatten_plus_one_hot = build_feature_matrix(
        "flatten_plus_one_hot",
        residue_embeddings,
        sequences=["ACD", "AAA"],
    )
    one_hot = build_feature_matrix(
        "one_hot_flat",
        residue_embeddings,
        sequences=["ACD", "AAA"],
    )
    mean_plus_one_hot = build_feature_matrix(
        "mean_pool_plus_one_hot",
        residue_embeddings,
        sequences=["ACD", "AAA"],
    )
    mean_l2_plus_one_hot = build_feature_matrix(
        "mean_pool_l2_plus_one_hot",
        residue_embeddings,
        sequences=["ACD", "AAA"],
    )
    mean_z_plus_one_hot = build_feature_matrix(
        "mean_pool_zscore_plus_one_hot",
        residue_embeddings,
        sequences=["ACD", "AAA"],
    )

    assert mean_pool.shape == (2, 4)
    assert mean_max.shape == (2, 8)
    assert flatten.shape == (2, 12)
    assert flatten_plus_one_hot.shape == (2, 72)
    assert one_hot.shape == (2, 60)
    assert mean_plus_one_hot.shape == (2, 64)
    assert mean_l2_plus_one_hot.shape == (2, 64)
    assert mean_z_plus_one_hot.shape == (2, 64)


def test_regression_metrics_cover_ranking_and_fit_quality():
    y_true = np.array([0.1, 0.2, 0.9, 1.0], dtype=float)
    y_pred = np.array([0.0, 0.3, 0.8, 0.95], dtype=float)

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["spearman"] > 0.7
    assert metrics["pearson"] > 0.8
    assert metrics["top10_overlap_rate"] == 1.0


def test_probe_selection_and_fit_work_on_small_dense_problem():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(24, 6))
    weights = np.array([2.0, -1.0, 0.0, 0.5, 0.0, 1.5])
    y = X @ weights + rng.normal(scale=0.05, size=24)

    spec = ProbeSpec(
        name="test_mean_pool_ridge",
        feature_name="mean_pool",
        estimator_name="ridge",
        search_grid=(0.01, 0.1, 1.0),
    )
    model, hyperparameter = fit_best_probe(spec, X, y, random_seed=0, scorer=spearmanr)
    predictions = model.predict(X)

    assert hyperparameter in spec.search_grid
    assert spearmanr(y, predictions) > 0.9


def test_missing_cache_can_be_filled_without_writing_cache(monkeypatch, tmp_path):
    loader = CachedTaskEmbeddingsLoader(
        model_name="fake-esm",
        max_length=None,
        cache_root=tmp_path / "esm_embeddings",
        compute_missing_embeddings=True,
    )
    fake_embedding = torch.ones((2, 3, 4), dtype=torch.float32)

    monkeypatch.setattr(
        loader,
        "_compute_residue_embeddings",
        lambda sequences: fake_embedding[: len(sequences)],
    )

    loaded = loader._load_sequences(["AAA", "CCC"], task="X", split_name="train")

    assert loaded.shape == (2, 3, 4)
    assert not list((tmp_path / "esm_embeddings").rglob("*.pt"))
