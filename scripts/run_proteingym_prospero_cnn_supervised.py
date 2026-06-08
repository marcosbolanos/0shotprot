#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from prospero.surrogate import CNN, sequence_to_one_hot
from prospero.utils import set_seed

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(ALPHABET)
CV_SCHEMES = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = list(sequences)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = sequence_to_one_hot(self.sequences[idx], ALPHABET).permute(1, 0).float()
        if self.labels is None:
            return x
        return x, torch.tensor(self.labels[idx], dtype=torch.float32)


def valid_sequence(seq):
    return isinstance(seq, str) and len(seq) > 4 and all(aa in AA_SET for aa in seq)


def get_fold_df(dms_id, dms_file, singles_dir, multiples_dir, scheme):
    singles_path = singles_dir / dms_file
    multiples_path = multiples_dir / dms_file
    if scheme == "fold_random_5" and multiples_path.exists():
        df = pd.read_csv(multiples_path)
        fold_col = "fold_rand_multiples"
    else:
        if not singles_path.exists():
            return None, None, "missing_fold_file"
        df = pd.read_csv(singles_path)
        fold_col = scheme
    if fold_col not in df.columns:
        return None, None, f"missing_{fold_col}"
    df = df[["mutant", "mutated_sequence", "DMS_score", fold_col]].copy()
    df = df.rename(columns={fold_col: "fold"})
    df = df[df["mutated_sequence"].map(valid_sequence)]
    df = df[np.isfinite(pd.to_numeric(df["DMS_score"], errors="coerce"))].copy()
    df["DMS_score"] = df["DMS_score"].astype(float)
    df["fold"] = df["fold"].astype(int)
    if df.empty:
        return None, None, "empty_after_filter"
    return df, fold_col, None


def train_one_model(train_seqs, train_y, valid_seqs, valid_y, seq_len, args, seed):
    set_seed(seed, False)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = CNN(num_input_channels=20, seq_length=seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(
        SequenceDataset(train_seqs, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = torch.utils.data.DataLoader(
        SequenceDataset(valid_seqs, valid_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    best_state = None
    best_loss = float("inf")
    patience = 0
    epochs_run = 0
    for epoch in range(1, args.max_epochs + 1):
        epochs_run = epoch
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % args.epochs_per_valid == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    pred = model(xb).squeeze(-1)
                    losses.append(float(loss_fn(pred, yb).detach().cpu()))
            val_loss = float(np.mean(losses)) if losses else float("inf")
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epochs": epochs_run, "best_valid_loss": best_loss}


@torch.no_grad()
def predict_model(model, sequences, args):
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(
        SequenceDataset(sequences),
        batch_size=args.predict_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    preds = []
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        preds.append(model(xb).squeeze(-1).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=float)


def score_fold(df, fold, args):
    train_df = df[df["fold"] != fold].copy()
    test_df = df[df["fold"] == fold].copy()
    if len(train_df) < args.min_train or len(test_df) < args.min_test:
        return None, {"skip": "small_split", "train": len(train_df), "test": len(test_df)}
    seq_len = len(train_df["mutated_sequence"].iloc[0])
    if any(len(s) != seq_len for s in train_df["mutated_sequence"]) or any(len(s) != seq_len for s in test_df["mutated_sequence"]):
        return None, {"skip": "variable_length"}
    if args.max_seq_len and seq_len > args.max_seq_len:
        return None, {"skip": "seq_len_cap", "seq_len": seq_len}
    y = train_df["DMS_score"].to_numpy(dtype=float)
    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std <= 1e-12:
        return None, {"skip": "zero_std"}
    y_norm = (y - y_mean) / y_std
    tr_seqs, va_seqs, tr_y, va_y = train_test_split(
        train_df["mutated_sequence"].to_numpy(),
        y_norm,
        test_size=args.valid_fraction,
        random_state=args.seed + fold,
    )
    all_preds = []
    model_stats = []
    for member in range(args.ensemble_size):
        model, stats = train_one_model(
            tr_seqs,
            tr_y,
            va_seqs,
            va_y,
            seq_len,
            args,
            seed=args.seed + 1009 * member + 17 * fold,
        )
        all_preds.append(predict_model(model, test_df["mutated_sequence"].tolist(), args))
        model_stats.append(stats)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pred = np.mean(np.stack(all_preds, axis=0), axis=0)
    out = test_df[["mutant", "mutated_sequence", "DMS_score"]].copy()
    out["labels_fitness"] = (test_df["DMS_score"].to_numpy(dtype=float) - y_mean) / y_std
    out["predictions_fitness"] = pred
    out["heldout_fold"] = fold
    meta = {
        "train": len(train_df),
        "test": len(test_df),
        "seq_len": seq_len,
        "label_mean": y_mean,
        "label_std": y_std,
        "model_stats": model_stats,
    }
    return out, meta


def score_assay(row, args, score_root):
    dms_id = row.DMS_id
    dms_file = row.DMS_filename
    singles_dir = Path(args.cv_singles_dir)
    multiples_dir = Path(args.cv_multiples_dir)
    assay_results = []
    assay_meta = {"DMS_id": dms_id, "DMS_filename": dms_file, "schemes": {}}
    for scheme in args.cv_schemes.split(","):
        scheme = scheme.strip()
        if not scheme:
            continue
        df, fold_col, err = get_fold_df(dms_id, dms_file, singles_dir, multiples_dir, scheme)
        if err:
            assay_meta["schemes"][scheme] = {"skip": err}
            continue
        if args.max_variants and len(df) > args.max_variants:
            assay_meta["schemes"][scheme] = {"skip": "variant_cap", "n_variants": int(len(df))}
            continue
        fold_outputs = []
        scheme_meta = {"fold_source_column": fold_col, "n_variants": int(len(df)), "folds": {}}
        start = time.perf_counter()
        for fold in range(5):
            out, meta = score_fold(df, fold, args)
            scheme_meta["folds"][str(fold)] = meta
            if out is not None:
                fold_outputs.append(out)
        scheme_meta["seconds"] = float(time.perf_counter() - start)
        if fold_outputs:
            pred_df = pd.concat(fold_outputs, ignore_index=True)
            pred_df = pred_df.sort_values(["heldout_fold", "mutant"]).reset_index(drop=True)
            out_dir = score_root / scheme / args.model_location
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(out_dir / f"{dms_id}.csv", index=False)
            sp = spearmanr(pred_df["labels_fitness"], pred_df["predictions_fitness"])[0]
            mse = mean_squared_error(pred_df["labels_fitness"], pred_df["predictions_fitness"])
            assay_results.append({
                "DMS_id": dms_id,
                "model_name": args.model_name,
                "fold_variable_name": scheme,
                "Spearman": float(sp),
                "MSE": float(mse),
                "n_scored": int(len(pred_df)),
            })
            scheme_meta["Spearman"] = float(sp)
            scheme_meta["MSE"] = float(mse)
            scheme_meta["n_scored"] = int(len(pred_df))
        else:
            scheme_meta["skip"] = "no_fold_outputs"
        assay_meta["schemes"][scheme] = scheme_meta
    return assay_results, assay_meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dms-reference", default="external/ProteinGym/reference_files/DMS_substitutions.csv")
    p.add_argument("--cv-singles-dir", default="external/ProteinGym_data/cv_folds_singles_substitutions")
    p.add_argument("--cv-multiples-dir", default="external/ProteinGym_data/cv_folds_multiples_substitutions")
    p.add_argument("--output-dir", default="outputs/proteingym_prospero_cnn_supervised_subs")
    p.add_argument("--model-name", default="ProSpero_CNN_ensemble")
    p.add_argument("--model-location", default="ProSpero_CNN_ensemble")
    p.add_argument("--cv-schemes", default=",".join(CV_SCHEMES))
    p.add_argument("--assay-ids", default=None, help="Comma-separated DMS IDs or name substrings to include")
    p.add_argument("--max-assays", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-variants", type=int, default=20000)
    p.add_argument("--max-seq-len", type=int, default=1200)
    p.add_argument("--min-train", type=int, default=20)
    p.add_argument("--min-test", type=int, default=5)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--max-epochs", type=int, default=3000)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--epochs-per-valid", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predict-batch-size", type=int, default=512)
    p.add_argument("--valid-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    out_root = Path(args.output_dir)
    score_root = out_root / "scores"
    out_root.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(args.dms_reference)
    if args.assay_ids:
        needles = [x.strip() for x in args.assay_ids.split(",") if x.strip()]
        mask = np.zeros(len(ref), dtype=bool)
        for needle in needles:
            mask |= ref["DMS_id"].astype(str).str.contains(needle, regex=False).to_numpy()
            mask |= ref["DMS_filename"].astype(str).str.contains(needle, regex=False).to_numpy()
        ref = ref[mask].copy()
    if args.num_shards > 1:
        ref = ref.iloc[np.arange(len(ref)) % args.num_shards == args.shard_index].copy()
    if args.max_assays is not None:
        ref = ref.head(args.max_assays).copy()

    all_results = []
    all_meta = []
    for idx, row in enumerate(ref.itertuples(index=False), 1):
        print(f"[{idx}/{len(ref)}] scoring {row.DMS_id} {row.DMS_filename}", flush=True)
        try:
            results, meta = score_assay(row, args, score_root)
        except Exception as exc:
            results = []
            meta = {"DMS_id": row.DMS_id, "DMS_filename": row.DMS_filename, "error": repr(exc)}
            print(f"ERROR {row.DMS_id}: {exc}", flush=True)
        all_results.extend(results)
        all_meta.append(meta)
        pd.DataFrame(all_results).to_csv(out_root / f"raw_scores_shard{args.shard_index}.csv", index=False)
        with (out_root / f"metadata_shard{args.shard_index}.jsonl").open("w") as h:
            for item in all_meta:
                h.write(json.dumps(item, sort_keys=True) + "\n")
    print("wrote", out_root / f"raw_scores_shard{args.shard_index}.csv")


if __name__ == "__main__":
    main()
