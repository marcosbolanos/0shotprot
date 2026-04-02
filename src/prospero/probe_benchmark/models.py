from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    feature_name: str
    estimator_name: str
    search_grid: tuple[float | int, ...]
    pca_components: int | None = None
    description: str = ""
    metric_to_optimize: str = "spearman"
    metadata: dict[str, str] = field(default_factory=dict)


DEFAULT_PROBE_SPECS: tuple[ProbeSpec, ...] = (
    ProbeSpec(
        name="mean_pool_ridge",
        feature_name="mean_pool",
        estimator_name="ridge",
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="Mean-pooled residues with ridge regression.",
    ),
    ProbeSpec(
        name="mean_max_ridge",
        feature_name="mean_max",
        estimator_name="ridge",
        search_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        description="Mean+max pooled residues with ridge regression.",
    ),
    ProbeSpec(
        name="flatten_ridge",
        feature_name="flatten",
        estimator_name="ridge",
        search_grid=(0.1, 1.0, 10.0, 100.0, 1000.0),
        description="Flattened per-residue features with ridge regression.",
    ),
    ProbeSpec(
        name="flatten_pca_ridge",
        feature_name="flatten",
        estimator_name="ridge",
        search_grid=(0.1, 1.0, 10.0, 100.0),
        pca_components=128,
        description="Flattened per-residue features, PCA compressed, then ridge.",
    ),
    ProbeSpec(
        name="mean_pool_knn",
        feature_name="mean_pool",
        estimator_name="knn",
        search_grid=(3, 5, 9, 15, 21),
        description="Mean-pooled residues with kNN regression.",
    ),
)


def _build_estimator(spec: ProbeSpec, hyperparameter: float | int, n_train: int) -> Pipeline:
    steps: list[tuple[str, object]] = [("scale", StandardScaler())]

    if spec.pca_components is not None:
        max_components = min(spec.pca_components, max(1, n_train - 1))
        steps.append(
            ("pca", PCA(n_components=max_components, svd_solver="auto", random_state=0))
        )

    if spec.estimator_name == "ridge":
        steps.append(("model", Ridge(alpha=float(hyperparameter))))
    elif spec.estimator_name == "knn":
        steps.append(
            (
                "model",
                KNeighborsRegressor(
                    n_neighbors=min(int(hyperparameter), n_train),
                    weights="distance",
                ),
            )
        )
    else:
        raise ValueError(f"Unsupported estimator_name={spec.estimator_name!r}")

    return Pipeline(steps)


def fit_best_probe(
    spec: ProbeSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int,
    scorer,
):
    if len(X_train) < 2:
        raise ValueError("Need at least 2 training samples to fit a probe.")

    if len(X_train) < 6:
        hyperparameter = spec.search_grid[0]
        model = _build_estimator(spec, hyperparameter, len(X_train))
        model.fit(X_train, y_train)
        return model, hyperparameter

    test_size = max(1, int(round(len(X_train) * 0.2)))
    test_size = min(test_size, len(X_train) - 1)
    X_fit, X_tune, y_fit, y_tune = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        random_state=random_seed,
    )

    best_score = None
    best_hyperparameter = None
    for hyperparameter in spec.search_grid:
        model = _build_estimator(spec, hyperparameter, len(X_fit))
        model.fit(X_fit, y_fit)
        predictions = model.predict(X_tune)
        score = float(scorer(y_tune, predictions))
        if best_score is None or score > best_score:
            best_score = score
            best_hyperparameter = hyperparameter

    final_model = _build_estimator(spec, best_hyperparameter, len(X_train))
    final_model.fit(X_train, y_train)
    return final_model, best_hyperparameter
