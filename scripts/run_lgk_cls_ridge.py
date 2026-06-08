from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from prospero.dataset import RegressionDataset


def normalize_sequences(sequences) -> list[str]:
    return ["".join(map(str, sequence)) for sequence in sequences]


def cache_key(sequences: list[str], model_name: str, max_length: int | None) -> str:
    payload = json.dumps(
        {
            "model_name": model_name,
            "max_length": max_length,
            "sequences": sequences,
            "representation": "cls_last_hidden_state_v1",
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_or_compute_cls_embeddings(
    *,
    sequences: list[str],
    model_name: str,
    max_length: int | None,
    batch_size: int,
    cache_dir: Path,
    device: str,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(sequences, model_name, max_length)
    cache_path = cache_dir / f"lgk_cls_{key}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["embeddings"].astype(np.float32, copy=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings: list[np.ndarray] = []

    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": max_length is not None,
    }
    if max_length is not None:
        tokenizer_kwargs["max_length"] = max_length

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            encoded = tokenizer(batch_sequences, **tokenizer_kwargs)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            cls_embedding = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.append(cls_embedding.astype(np.float32, copy=False))

    matrix = np.concatenate(embeddings, axis=0)
    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        embeddings=matrix,
        model_name=np.array(model_name),
        max_length=np.array(-1 if max_length is None else max_length),
    )
    tmp_path.replace(cache_path)
    return matrix


def evaluate_model(
    *,
    name: str,
    train_features: np.ndarray,
    train_scores: np.ndarray,
    valid_features: np.ndarray,
    valid_scores: np.ndarray,
    ridge_alpha: float,
    fit_intercept: bool,
) -> dict[str, object]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features)
    x_valid = scaler.transform(valid_features)

    model = Ridge(alpha=ridge_alpha, fit_intercept=fit_intercept)
    model.fit(x_train, train_scores)
    valid_pred = model.predict(x_valid).astype(np.float32)
    train_pred = model.predict(x_train).astype(np.float32)
    spearman = spearmanr(valid_scores, valid_pred).correlation

    return {
        "name": name,
        "n_train": int(len(train_scores)),
        "train_r2": float(r2_score(train_scores, train_pred)),
        "valid_r2": float(r2_score(valid_scores, valid_pred)),
        "valid_rmse": float(np.sqrt(mean_squared_error(valid_scores, valid_pred))),
        "valid_mae": float(mean_absolute_error(valid_scores, valid_pred)),
        "valid_spearman": None if np.isnan(spearman) else float(spearman),
        "valid_predictions": valid_pred.tolist(),
    }


def write_predictions(
    *,
    path: Path,
    valid_sequences: list[str],
    valid_scores: np.ndarray,
    results: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_by_name = {
        str(result["name"]): result["valid_predictions"] for result in results
    }
    fieldnames = ["sequence", "oracle_score", *prediction_by_name.keys()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sequence in enumerate(valid_sequences):
            row = {
                "sequence": sequence,
                "oracle_score": float(valid_scores[idx]),
            }
            for name, predictions in prediction_by_name.items():
                row[name] = float(predictions[idx])
            writer.writerow(row)


def parse_top_k_values(raw_values: str) -> list[int]:
    values: list[int] = []
    for raw_value in raw_values.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        value = int(raw_value)
        if value <= 0:
            raise ValueError("--top-k-values must contain positive integers")
        if value not in values:
            values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LGK ridge models on ESM CLS embeddings."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-fit-intercept", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/embedding_cache"))
    parser.add_argument(
        "--top-k-values",
        default="10",
        help="Comma-separated top-k training subset sizes to evaluate.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RegressionDataset("LGK")
    train_sequences = normalize_sequences(dataset.train)
    valid_sequences = normalize_sequences(dataset.valid)
    train_scores = np.asarray(dataset.train_scores, dtype=np.float32)
    valid_scores = np.asarray(dataset.valid_scores, dtype=np.float32)

    all_sequences = list(dict.fromkeys([*train_sequences, *valid_sequences]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = load_or_compute_cls_embeddings(
        sequences=all_sequences,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        device=device,
    )
    embedding_by_sequence = dict(zip(all_sequences, embeddings))
    train_features = np.stack([embedding_by_sequence[seq] for seq in train_sequences])
    valid_features = np.stack([embedding_by_sequence[seq] for seq in valid_sequences])

    sorted_train_indices = np.argsort(train_scores)[::-1]
    top_k_values = parse_top_k_values(args.top_k_values)
    results = [
        evaluate_model(
            name="full_train",
            train_features=train_features,
            train_scores=train_scores,
            valid_features=valid_features,
            valid_scores=valid_scores,
            ridge_alpha=args.ridge_alpha,
            fit_intercept=args.ridge_fit_intercept,
        )
    ]
    top_k_train_scores: dict[str, list[float]] = {}
    for top_k in top_k_values:
        if top_k > len(train_scores):
            raise ValueError(
                f"Requested top-k={top_k}, but LGK train set has {len(train_scores)} rows"
            )
        top_k_indices = sorted_train_indices[:top_k]
        top_k_train_scores[f"top{top_k}_train"] = train_scores[top_k_indices].tolist()
        results.append(
            evaluate_model(
                name=f"top{top_k}_train",
                train_features=train_features[top_k_indices],
                train_scores=train_scores[top_k_indices],
                valid_features=valid_features,
                valid_scores=valid_scores,
                ridge_alpha=args.ridge_alpha,
                fit_intercept=args.ridge_fit_intercept,
            )
        )

    predictions_path = args.output_dir / "valid_predictions.csv"
    write_predictions(
        path=predictions_path,
        valid_sequences=valid_sequences,
        valid_scores=valid_scores,
        results=results,
    )

    for result in results:
        result.pop("valid_predictions", None)

    summary = {
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "LGK",
        "model_name": args.model_name,
        "representation": "cls_last_hidden_state_v1",
        "device": device,
        "cache_dir": str(args.cache_dir),
        "n_train_full": int(len(train_scores)),
        "n_valid": int(len(valid_scores)),
        "top_k_values": top_k_values,
        "top_k_train_scores": top_k_train_scores,
        "results": results,
        "predictions_csv": str(predictions_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {summary_path}")
    print(f"saved: {predictions_path}")


if __name__ == "__main__":
    main()
