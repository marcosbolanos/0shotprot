from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MLPHyperparameter = tuple[int, ...]
ProbeHyperparameter = float | int | MLPHyperparameter


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    feature_name: str
    estimator_name: str
    search_grid: tuple[ProbeHyperparameter, ...]
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


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_layers: tuple[int, ...],
        random_seed: int,
        learning_rate: float = 1e-3,
        l2_lambda: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 64,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.model_: nn.Module | None = None

    def _build_network(self, n_features: int) -> nn.Module:
        layers: list[nn.Module] = []
        in_features = n_features
        for width in self.hidden_layers:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        n_samples = len(X_arr)
        if n_samples < 2:
            raise ValueError("Need at least 2 training samples to fit MLP probe.")

        torch.manual_seed(self.random_seed)
        rng = np.random.RandomState(self.random_seed)

        if n_samples >= 6:
            test_size = max(1, int(round(n_samples * 0.2)))
            test_size = min(test_size, n_samples - 1)
            X_fit, X_eval, y_fit, y_eval = train_test_split(
                X_arr,
                y_arr,
                test_size=test_size,
                random_state=self.random_seed,
            )
        else:
            X_fit, X_eval = X_arr, X_arr
            y_fit, y_eval = y_arr, y_arr

        self.model_ = self._build_network(X_arr.shape[1])
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        loss_fn = nn.MSELoss()

        X_fit_t = torch.from_numpy(X_fit)
        y_fit_t = torch.from_numpy(y_fit).reshape(-1, 1)
        X_eval_t = torch.from_numpy(X_eval)
        y_eval_t = torch.from_numpy(y_eval).reshape(-1, 1)

        best_eval_loss = float("inf")
        best_state = copy.deepcopy(self.model_.state_dict())
        stale_epochs = 0

        for _ in range(self.max_epochs):
            self.model_.train()
            order = rng.permutation(len(X_fit_t))
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = X_fit_t[idx]
                yb = y_fit_t[idx]
                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                eval_pred = self.model_(X_eval_t)
                eval_loss = float(loss_fn(eval_pred, y_eval_t).item())
            if eval_loss < best_eval_loss - 1e-12:
                best_eval_loss = eval_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        self.model_.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("MLP model is not fit yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(torch.from_numpy(X_arr)).reshape(-1)
        return predictions.cpu().numpy().astype(float)


def _build_estimator(
    spec: ProbeSpec,
    hyperparameter: ProbeHyperparameter,
    n_train: int,
    random_seed: int,
) -> Pipeline:
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
    elif spec.estimator_name == "mlp":
        if not isinstance(hyperparameter, tuple) or not hyperparameter:
            raise ValueError(
                f"MLP hyperparameter must be a non-empty tuple, got {hyperparameter!r}."
            )
        hidden_layers = tuple(int(width) for width in hyperparameter)
        if any(width < 1 for width in hidden_layers):
            raise ValueError(f"MLP hidden layer widths must be positive, got {hidden_layers!r}.")
        steps.append(
            (
                "model",
                TorchMLPRegressor(
                    hidden_layers=hidden_layers,
                    random_seed=random_seed,
                    l2_lambda=1e-4,
                    patience=10,
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
        model = _build_estimator(spec, hyperparameter, len(X_train), random_seed=random_seed)
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
    best_hyperparameter: ProbeHyperparameter | None = None
    for hyperparameter in spec.search_grid:
        model = _build_estimator(spec, hyperparameter, len(X_fit), random_seed=random_seed)
        model.fit(X_fit, y_fit)
        predictions = model.predict(X_tune)
        score = float(scorer(y_tune, predictions))
        if best_score is None or score > best_score:
            best_score = score
            best_hyperparameter = hyperparameter

    if best_hyperparameter is None:
        raise RuntimeError(f"Search grid is empty for probe {spec.name!r}.")
    final_model = _build_estimator(
        spec,
        best_hyperparameter,
        len(X_train),
        random_seed=random_seed,
    )
    final_model.fit(X_train, y_train)
    return final_model, best_hyperparameter
