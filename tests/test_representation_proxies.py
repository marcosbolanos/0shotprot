import csv

import numpy as np
import torch

from prospero.probe_benchmark.features import CachedTaskEmbeddingsLoader, build_feature_matrix
from prospero.probe_benchmark.interplm import InterPLMSparseAutoencoder
from prospero.probe_benchmark.interplm_pipeline import (
    InterPLMBenchmarkRunner,
    build_interplm_probe_specs,
)
from prospero.probe_benchmark.metrics import regression_metrics, spearmanr
from prospero.probe_benchmark.models import (
    ProbeSpec,
    extract_linear_probe_parameters,
    fit_best_probe,
)


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
    fit_result = fit_best_probe(spec, X, y, random_seed=0, scorer=spearmanr)
    predictions = fit_result.model.predict(X)

    assert fit_result.selected_hyperparameter in spec.search_grid
    assert spearmanr(y, predictions) > 0.9


def test_probe_selection_and_fit_work_for_mlp():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(36, 8))
    y = (
        0.8 * X[:, 0]
        - 0.6 * X[:, 1]
        + 0.4 * np.maximum(X[:, 2], 0.0)
        - 0.2 * np.maximum(-X[:, 3], 0.0)
        + rng.normal(scale=0.05, size=36)
    )

    spec = ProbeSpec(
        name="test_mean_pool_mlp",
        feature_name="mean_pool",
        estimator_name="mlp",
        search_grid=((4,), (8,), (4, 4)),
    )
    fit_result = fit_best_probe(spec, X, y, random_seed=0, scorer=spearmanr)
    predictions = fit_result.model.predict(X)

    assert fit_result.selected_hyperparameter in spec.search_grid
    assert spearmanr(y, predictions) > 0.5


def test_lasso_probe_reports_alpha_diagnostics():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(30, 5))
    y = 1.5 * X[:, 0] - 0.8 * X[:, 3] + rng.normal(scale=0.05, size=30)

    spec = ProbeSpec(
        name="test_mean_pool_lasso",
        feature_name="mean_pool",
        estimator_name="lasso",
        search_grid=(1.0, 0.1, 0.01),
        metadata={"hyperparameter_scale": "alpha_over_alpha_max"},
    )
    fit_result = fit_best_probe(spec, X, y, random_seed=0, scorer=spearmanr)

    assert fit_result.selected_search_value in spec.search_grid
    assert fit_result.selected_hyperparameter > 0.0
    assert len(fit_result.hyperparameter_diagnostics) == len(spec.search_grid)
    for diagnostic in fit_result.hyperparameter_diagnostics:
        assert diagnostic["search_value"] in spec.search_grid
        assert diagnostic["resolved_hyperparameter"] > 0.0
        assert diagnostic["alpha_max"] is not None
        assert isinstance(diagnostic["score"], float)


def test_extract_linear_probe_parameters_match_pipeline_predictions():
    rng = np.random.RandomState(3)
    X = rng.normal(size=(32, 5))
    y = 1.2 * X[:, 0] - 0.7 * X[:, 2] + 0.4 * X[:, 4] + rng.normal(scale=0.01, size=32)

    spec = ProbeSpec(
        name="test_mean_pool_ridge",
        feature_name="mean_pool",
        estimator_name="ridge",
        search_grid=(0.01, 0.1, 1.0),
    )
    fit_result = fit_best_probe(spec, X, y, random_seed=0, scorer=spearmanr)
    parameters = extract_linear_probe_parameters(fit_result.model)

    assert parameters is not None
    manual_predictions = X @ parameters.input_coefficients + parameters.input_intercept
    np.testing.assert_allclose(
        manual_predictions,
        fit_result.model.predict(X),
        rtol=1e-6,
        atol=1e-6,
    )


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


def test_build_interplm_probe_specs_includes_onehot_and_requested_layers():
    specs = build_interplm_probe_specs((2, 4))

    assert [spec.name for spec in specs] == [
        "one_hot_ridge",
        "interplm_l2_ridge",
        "interplm_l4_ridge",
    ]
    assert specs[1].feature_name == "interplm_l2_mean_pool"
    assert specs[2].metadata["interplm_layer"] == "4"


def test_build_interplm_probe_specs_support_lasso_and_elasticnet():
    lasso_specs = build_interplm_probe_specs((2,), estimator_name="lasso")
    elasticnet_specs = build_interplm_probe_specs((2,), estimator_name="elasticnet")

    assert [spec.name for spec in lasso_specs] == ["one_hot_lasso", "interplm_l2_lasso"]
    assert lasso_specs[0].metadata["hyperparameter_scale"] == "alpha_over_alpha_max"
    assert [spec.name for spec in elasticnet_specs] == [
        "one_hot_elasticnet",
        "interplm_l2_elasticnet",
    ]
    assert elasticnet_specs[1].metadata["l1_ratio"] == "0.9"


def test_interplm_mean_pool_matches_manual_chunked_encoding(tmp_path):
    loader = object.__new__(InterPLMBenchmarkRunner)
    del loader  # placate linters for the object.__new__ smoke.

    from prospero.probe_benchmark.interplm_pipeline import InterPLMTaskFeatureLoader

    feature_loader = InterPLMTaskFeatureLoader(
        model_name="fake-esm",
        max_length=None,
        cache_root=tmp_path / "interplm_cache",
        interplm_layers=(1,),
        compute_missing_features=False,
        sae_token_chunk_size=2,
    )
    sae = InterPLMSparseAutoencoder(input_dim=3, feature_dim=4)
    with torch.no_grad():
        sae.bias.zero_()
        sae.encoder.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
        sae.encoder.bias.zero_()

    residue_embeddings = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]],
            [[2.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    pooled = feature_loader._mean_pool_sae_activations(sae, residue_embeddings)
    manual = torch.relu(sae.encoder(residue_embeddings)).mean(dim=1)

    assert torch.allclose(pooled, manual)


def test_interplm_benchmark_runner_writes_summary_and_plots(monkeypatch, tmp_path):
    feature_names = ["one_hot_flat", "interplm_l1_mean_pool", "interplm_l2_mean_pool"]
    train_features = {
        name: np.arange(24, dtype=np.float32).reshape(6, 4) + idx
        for idx, name in enumerate(feature_names)
    }
    valid_features = {
        name: np.arange(12, dtype=np.float32).reshape(3, 4) + idx
        for idx, name in enumerate(feature_names)
    }

    class FakeLoader:
        model_name = "fake-esm"
        max_length = None

        def load_task(self, task: str):
            from prospero.probe_benchmark.interplm_pipeline import TaskFeatureSet

            return TaskFeatureSet(
                task=task,
                sequence_length=3,
                train_sequences=["AAA"] * 6,
                valid_sequences=["AAA"] * 3,
                train_scores=np.linspace(0.0, 1.0, 6, dtype=float),
                valid_scores=np.linspace(0.0, 1.0, 3, dtype=float),
                train_feature_matrices=train_features,
                valid_feature_matrices=valid_features,
            )

    runner = InterPLMBenchmarkRunner(
        tasks=["ToyTask"],
        budgets=[2, 4],
        seeds=[1],
        output_dir=tmp_path / "interplm_outputs",
        interplm_layers=(1, 2),
        require_cache=False,
        compute_missing_features=False,
    )
    monkeypatch.setattr(runner, "loader", FakeLoader())

    summary = runner.run()

    assert summary["metadata"]["interplm_layers"] == [1, 2]
    assert set(summary["tasks"]["ToyTask"]["configs"]) == {
        "one_hot_ridge",
        "interplm_l1_ridge",
        "interplm_l2_ridge",
    }
    assert (tmp_path / "interplm_outputs" / "benchmark_results.json").exists()
    coefficient_index_path = tmp_path / "interplm_outputs" / "coefficient_index.csv"
    assert coefficient_index_path.exists()
    with open(coefficient_index_path, "r", encoding="utf-8", newline="") as handle:
        coefficient_rows = list(csv.DictReader(handle))
    assert len(coefficient_rows) == 6
    coefficient_artifact = tmp_path / "interplm_outputs" / coefficient_rows[0]["coefficient_path"]
    with np.load(coefficient_artifact) as saved:
        assert saved["coefficients"].shape == (4,)
        assert saved["train_indices"].shape == (2,)
    assert (
        tmp_path / "interplm_outputs" / "summary_plots" / "cross_task_mean_spearman_by_budget.png"
    ).exists()
    assert (
        tmp_path / "interplm_outputs" / "summary_plots" / "per_task_spearman_by_budget.png"
    ).exists()


def test_lasso_benchmark_runner_writes_alpha_diagnostics(monkeypatch, tmp_path):
    feature_names = ["one_hot_flat", "interplm_l1_mean_pool"]
    train_features = {
        name: np.arange(40, dtype=np.float32).reshape(10, 4) + idx
        for idx, name in enumerate(feature_names)
    }
    valid_features = {
        name: np.arange(16, dtype=np.float32).reshape(4, 4) + idx
        for idx, name in enumerate(feature_names)
    }

    class FakeLoader:
        model_name = "fake-esm"
        max_length = None

        def load_task(self, task: str):
            from prospero.probe_benchmark.interplm_pipeline import TaskFeatureSet

            return TaskFeatureSet(
                task=task,
                sequence_length=3,
                train_sequences=["AAA"] * 10,
                valid_sequences=["AAA"] * 4,
                train_scores=np.linspace(0.0, 1.0, 10, dtype=float),
                valid_scores=np.linspace(0.0, 1.0, 4, dtype=float),
                train_feature_matrices=train_features,
                valid_feature_matrices=valid_features,
            )

    runner = InterPLMBenchmarkRunner(
        tasks=["ToyTask"],
        budgets=[8],
        seeds=[1, 2],
        output_dir=tmp_path / "interplm_lasso_outputs",
        interplm_layers=(1,),
        require_cache=False,
        compute_missing_features=False,
        estimator_name="lasso",
    )
    monkeypatch.setattr(runner, "loader", FakeLoader())

    summary = runner.run()

    assert summary["metadata"]["estimator_name"] == "lasso"
    assert summary["tasks"]["ToyTask"]["configs"]["interplm_l1_lasso"]["estimator_name"] == "lasso"
    assert (tmp_path / "interplm_lasso_outputs" / "alpha_diagnostics.csv").exists()
    coefficient_index_path = tmp_path / "interplm_lasso_outputs" / "coefficient_index.csv"
    assert coefficient_index_path.exists()
    with open(coefficient_index_path, "r", encoding="utf-8", newline="") as handle:
        coefficient_rows = list(csv.DictReader(handle))
    assert len(coefficient_rows) == 4
    assert {row["estimator_name"] for row in coefficient_rows} == {"lasso"}
    assert (
        tmp_path / "interplm_lasso_outputs" / "supplementary_plots" / "ToyTask_alpha_sweep.png"
    ).exists()


def test_interplm_feature_cache_can_be_built_from_cached_residue_embeddings(monkeypatch, tmp_path):
    from prospero.probe_benchmark.interplm_pipeline import InterPLMTaskFeatureLoader

    loader = InterPLMTaskFeatureLoader(
        model_name="fake-esm",
        max_length=None,
        cache_root=tmp_path / "interplm_cache",
        interplm_layers=(1,),
        require_cache=True,
        compute_missing_features=False,
        device="cpu",
    )
    residue_embedding = torch.tensor(
        [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
        dtype=torch.float32,
    )
    loader.residue_embedding_caches[1].set_many(["AAA"], [residue_embedding])

    sae = InterPLMSparseAutoencoder(input_dim=3, feature_dim=2)
    with torch.no_grad():
        sae.bias.zero_()
        sae.encoder.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        sae.encoder.bias.zero_()
    monkeypatch.setattr(loader, "_get_sae", lambda layer: sae)

    features = loader._load_feature_matrices(["AAA"], task="Toy", split_name="train")

    expected = torch.tensor([[0.75, 1.5]], dtype=torch.float32).numpy()
    np.testing.assert_allclose(features["interplm_l1_mean_pool"], expected)
    cached_features, missing = loader.feature_caches[1].get_many(["AAA"])
    assert not missing
    assert tuple(cached_features["AAA"].shape) == (2,)
