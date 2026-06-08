from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prospero.dataset import RegressionDataset
from prospero.experiments_config import WT_SEQUENCES
from prospero.plotting_style import COLORS, set_prospero_style

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I"]

MODELS = {
    "prosst": {
        "label": "ProSST",
        "root": ROOT / "outputs/prosst_zero_shot_alignment_20260603/predictive_scores",
        "score_col": "prosst_log_odds",
        "color": COLORS["prosst"],
        "axis": "Delta PLL / log-odds vs WT",
    },
    "esm2_650m": {
        "label": "ESM-2 650M",
        "root": ROOT / "outputs/esm2_650m_zero_shot_alignment_20260603/predictive_scores",
        "score_col": "masked_marginals",
        "color": COLORS["esm2"],
        "axis": "Delta PLL / masked marginal log-odds vs WT",
    },
    "evodiff": {
        "label": "EvoDiff",
        "root": ROOT / "outputs/evodiff_zero_shot_alignment_20260602/predictive_scores",
        "score_col": "masked_marginals",
        "color": COLORS["evodiff"],
        "axis": "Delta PLL / masked marginal log-odds vs WT",
    },
}


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def wt_fitness(task: str) -> tuple[float, str]:
    wt = WT_SEQUENCES[task]
    ds = RegressionDataset(task)
    vals = []
    sources = []
    for split, seqs, scores in [("train", ds.train, ds.train_scores), ("valid", ds.valid, ds.valid_scores)]:
        for seq, score in zip(seqs, scores):
            seq = seq if isinstance(seq, str) else "".join(map(str, seq))
            if seq == wt:
                vals.append(float(score))
                sources.append(split)
    if vals:
        return float(np.mean(vals)), "+".join(sorted(set(sources)))
    return 0.0, "fallback_zero"


def load_model_task(model_key: str, task: str) -> tuple[pd.DataFrame | None, dict]:
    spec = MODELS[model_key]
    path = spec["root"] / f"{task}.csv"
    if not path.exists():
        return None, {"task": task, "model": model_key, "status": "missing_csv"}
    df = pd.read_csv(path)
    score_col = spec["score_col"]
    if score_col not in df.columns:
        return None, {"task": task, "model": model_key, "status": f"missing_{score_col}"}
    center, center_source = wt_fitness(task)
    df = df.copy()
    df["delta_fitness"] = df["fitness"].astype(float) - center
    df["delta_pll"] = df[score_col].astype(float)
    df = df[np.isfinite(df["delta_fitness"]) & np.isfinite(df["delta_pll"])]
    rho = safe_corr(rankdata(df["delta_pll"].to_numpy()), rankdata(df["delta_fitness"].to_numpy()))
    pearson = safe_corr(df["delta_pll"].to_numpy(), df["delta_fitness"].to_numpy())
    summary = {
        "task": task,
        "model": model_key,
        "n": int(len(df)),
        "wt_fitness": center,
        "wt_fitness_source": center_source,
        "spearman": rho,
        "pearson": pearson,
        "delta_fitness_min": float(df["delta_fitness"].min()) if len(df) else None,
        "delta_fitness_max": float(df["delta_fitness"].max()) if len(df) else None,
        "delta_pll_min": float(df["delta_pll"].min()) if len(df) else None,
        "delta_pll_max": float(df["delta_pll"].max()) if len(df) else None,
    }
    return df, summary


def binned_trend(x: np.ndarray, y: np.ndarray, bins: int = 18):
    if len(x) < bins:
        return None
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 4:
        return None
    xs, ys, lo, hi = [], [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == edges[-1]:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        vals = y[mask]
        xv = x[mask]
        if len(vals) < 5:
            continue
        xs.append(float(np.median(xv)))
        ys.append(float(np.median(vals)))
        lo.append(float(np.quantile(vals, 0.25)))
        hi.append(float(np.quantile(vals, 0.75)))
    if len(xs) < 3:
        return None
    return np.asarray(xs), np.asarray(ys), np.asarray(lo), np.asarray(hi)


def symmetric_limits(values, q=0.995):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (-1.0, 1.0)
    lo, hi = np.quantile(values, [1 - q, q])
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = 0.05 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def plot_model(model_key: str, out_dir: Path, dpi: int):
    spec = MODELS[model_key]
    payloads = []
    summaries = []
    for task in TASKS:
        df, summary = load_model_task(model_key, task)
        summaries.append(summary)
        if df is not None and len(df):
            payloads.append((task, df, summary))
    if not payloads:
        return [], summaries

    fig, axes = plt.subplots(2, 4, figsize=(15.8, 8.7), sharex=False, sharey=False)
    for idx, task in enumerate(TASKS):
        ax = axes[idx // 4, idx % 4]
        item = next((p for p in payloads if p[0] == task), None)
        if item is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes, color=COLORS["muted"])
            ax.set_axis_off()
            continue
        _, df, summary = item
        x = df["delta_pll"].to_numpy(dtype=float)
        y = df["delta_fitness"].to_numpy(dtype=float)
        ylim = symmetric_limits(y, q=0.995)
        ax.axhline(0, color=COLORS["grid"], linewidth=1.0, zorder=0)
        ax.axvline(0, color=COLORS["grid"], linewidth=1.0, zorder=0)
        ax.scatter(x, y, s=12, alpha=1.0, color=spec["color"], edgecolors="none", rasterized=True)
        ax.set_title(task, loc="left", pad=4)
        ax.text(
            0.98,
            0.05,
            f"rho={summary['spearman']:.2f}\nn={summary['n']:,}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            color=COLORS["ink"],
        )
        ax.set_xlim(symmetric_limits(x, q=0.995))
        ax.set_ylim(ylim)
        ax.grid(True, axis="both")
        if idx % 4 == 0:
            ax.set_ylabel("Delta fitness (oracle - WT)")
        if idx // 4 == 1:
            ax.set_xlabel(spec["axis"])
    fig.suptitle(
        f"{spec['label']} zero-shot alignment across ProSpero landscapes",
        fontsize=20,
        fontweight=600,
        y=1.02,
    )
    fig.text(
        0.5,
        0.975,
        "Each point is a validation sequence. Axes use robust per-landscape limits.",
        ha="center",
        va="top",
        fontsize=13,
        color=COLORS["muted"],
        fontweight=300,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = out_dir / f"{model_key}_delta_pll_vs_delta_fitness.png"
    pdf = out_dir / f"{model_key}_delta_pll_vs_delta_fitness.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf], summaries


def plot_metric_summary(all_summaries: list[dict], out_dir: Path, dpi: int):
    rows = [s for s in all_summaries if s.get("status") is None and s.get("spearman") is not None]
    if not rows:
        return []
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12.8, 5.3))
    model_order = [k for k in MODELS if k in set(df["model"])]
    offsets = np.linspace(-0.24, 0.24, len(model_order)) if len(model_order) > 1 else [0]
    xbase = np.arange(len(TASKS))
    width = 0.18 if len(model_order) > 2 else 0.24
    for offset, model_key in zip(offsets, model_order):
        spec = MODELS[model_key]
        vals = []
        for task in TASKS:
            sub = df[(df["model"] == model_key) & (df["task"] == task)]
            vals.append(float(sub["spearman"].iloc[0]) if len(sub) else np.nan)
        ax.bar(xbase + offset, vals, width=width, label=spec["label"], color=spec["color"], alpha=0.92)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8, alpha=0.45)
    ax.set_xticks(xbase)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Spearman rho(delta PLL, delta fitness)")
    ax.set_title("Zero-shot alignment strength by landscape", loc="left", fontsize=18, fontweight=600)
    ax.grid(True, axis="y")
    ax.legend(ncol=len(model_order), loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.set_ylim(-0.15, 0.85)
    fig.tight_layout()
    png = out_dir / "zero_shot_alignment_spearman_summary.png"
    pdf = out_dir / "zero_shot_alignment_spearman_summary.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot delta PLL vs delta fitness for zero-shot prediction outputs.")
    parser.add_argument("--output_dir", default="outputs/zero_shot_alignment_plots_20260604")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def main():
    args = parse_args()
    set_prospero_style()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_written = []
    all_summaries = []
    for model_key in args.models:
        written, summaries = plot_model(model_key, out_dir, args.dpi)
        all_written.extend(written)
        all_summaries.extend(summaries)
    all_written.extend(plot_metric_summary(all_summaries, out_dir, args.dpi))
    summary_path = out_dir / "zero_shot_alignment_plot_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"written": [str(p) for p in all_written], "summaries": all_summaries}, handle, indent=2, sort_keys=True)
    print(f"Wrote {len(all_written)} plot files")
    print(summary_path)
    for path in all_written:
        print(path)


if __name__ == "__main__":
    main()
