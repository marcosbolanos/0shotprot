from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from prospero.dataset import RegressionDataset
from prospero.esm_lora import apply_lora_to_esm_layer, save_lora_adapter, truncate_esm_encoder
from prospero.surrogate import normalize_sequences


class SequenceScoreDataset(Dataset):
    def __init__(self, sequences: list[str], scores: np.ndarray) -> None:
        self.sequences = sequences
        self.scores = torch.as_tensor(scores, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return self.sequences[index], self.scores[index]


class CLSRegressionHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, cls_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(cls_embeddings).squeeze(-1)


def select_top_k(
    sequences: list[str],
    scores: np.ndarray,
    top_k: int | None,
) -> tuple[list[str], np.ndarray]:
    if top_k is None:
        return sequences, scores
    if top_k <= 0:
        raise ValueError("--initial-train-top-k must be positive")
    if top_k > len(scores):
        raise ValueError(f"Requested top_k={top_k}, but train set has {len(scores)}")
    indices = np.argsort(scores)[::-1][:top_k]
    return [sequences[int(index)] for index in indices], scores[indices]


def evaluate(
    *,
    model,
    head,
    tokenizer,
    loader,
    device: str,
    hidden_state_layer: int,
    max_length: int | None,
) -> dict[str, float]:
    model.eval()
    head.eval()
    predictions: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for sequences, scores in loader:
            encoded = tokenizer(
                list(sequences),
                return_tensors="pt",
                padding=True,
                truncation=max_length is not None,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True)
            cls = output.hidden_states[hidden_state_layer][:, 0, :]
            pred = head(cls)
            predictions.extend(pred.detach().cpu().tolist())
            labels.extend(scores.detach().cpu().tolist())
    pred_arr = np.asarray(predictions, dtype=np.float32)
    label_arr = np.asarray(labels, dtype=np.float32)
    spearman = spearmanr(label_arr, pred_arr).correlation
    return {
        "rmse": float(np.sqrt(mean_squared_error(label_arr, pred_arr))),
        "r2": float(r2_score(label_arr, pred_arr)),
        "spearman": None if np.isnan(spearman) else float(spearman),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune truncated ESM layer-CLS LoRA adapter for regression."
    )
    parser.add_argument("--task", default="AAV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--hidden-state-layer", type=int, default=2)
    parser.add_argument(
        "--lora-encoder-layer-index",
        type=int,
        default=None,
        help=(
            "Zero-based encoder block to LoRA. Defaults to hidden_state_layer - 1, "
            "which affects the repo convention hidden_states[layer]."
        ),
    )
    parser.add_argument("--target-modules", default="attention.self.query,attention.self.value")
    parser.add_argument("--initial-train-top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=142857)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RegressionDataset(args.task)
    train_sequences = normalize_sequences(dataset.train)
    valid_sequences = normalize_sequences(dataset.valid)
    train_scores = np.asarray(dataset.train_scores, dtype=np.float32)
    valid_scores = np.asarray(dataset.valid_scores, dtype=np.float32)
    train_sequences, train_scores = select_top_k(
        train_sequences,
        train_scores,
        args.initial_train_top_k,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    truncate_esm_encoder(model, hidden_state_layer=args.hidden_state_layer)
    for parameter in model.parameters():
        parameter.requires_grad = False

    lora_layer = (
        args.hidden_state_layer - 1
        if args.lora_encoder_layer_index is None
        else args.lora_encoder_layer_index
    )
    target_modules = tuple(
        item.strip() for item in args.target_modules.split(",") if item.strip()
    )
    replaced = apply_lora_to_esm_layer(
        model,
        encoder_layer_index=lora_layer,
        target_modules=target_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    head = CLSRegressionHead(model.config.hidden_size).to(device)

    trainable = [
        parameter for parameter in list(model.parameters()) + list(head.parameters())
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        SequenceScoreDataset(train_sequences, train_scores),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        SequenceScoreDataset(valid_sequences, valid_scores),
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        head.train()
        total_loss = 0.0
        n_seen = 0
        for sequences, scores in train_loader:
            encoded = tokenizer(
                list(sequences),
                return_tensors="pt",
                padding=True,
                truncation=args.max_length is not None,
                max_length=args.max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            scores = scores.to(device)
            output = model(**encoded, output_hidden_states=True)
            cls = output.hidden_states[args.hidden_state_layer][:, 0, :]
            predictions = head(cls)
            loss = loss_fn(predictions, scores)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(sequences)
            n_seen += len(sequences)

        valid_metrics = evaluate(
            model=model,
            head=head,
            tokenizer=tokenizer,
            loader=valid_loader,
            device=device,
            hidden_state_layer=args.hidden_state_layer,
            max_length=args.max_length,
        )
        record = {
            "epoch": epoch,
            "train_mse": total_loss / max(1, n_seen),
            "valid": valid_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        if valid_metrics["rmse"] < best_rmse:
            best_rmse = valid_metrics["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            metadata = {
                "task": args.task,
                "model_name": args.model_name,
                "hidden_state_layer": args.hidden_state_layer,
                "lora_encoder_layer_index": lora_layer,
                "target_modules": target_modules,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "initial_train_top_k": args.initial_train_top_k,
                "n_train": len(train_sequences),
                "n_valid": len(valid_sequences),
                "best_epoch": best_epoch,
                "best_valid": valid_metrics,
                "replaced_modules": replaced,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            save_lora_adapter(output_dir=args.output_dir, model=model, metadata=metadata)
            torch.save(head.state_dict(), args.output_dir / "regression_head.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    summary = {
        "best_epoch": best_epoch,
        "best_rmse": best_rmse,
        "history": history,
        "adapter_dir": str(args.output_dir),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
