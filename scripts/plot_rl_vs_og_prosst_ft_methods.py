import argparse
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I"]
BUDGETS = [8, 128]

OG_ROOTS = {
    "AAV": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/AAV_cnn",
    "LGK": ROOT / "outputs/out_240226_lgk_cnn",
    "GFP": ROOT / "outputs/out_240226_gfp_cnn",
    "Pab1": ROOT / "outputs/out_240226_pab1_cnn",
    "AMIE": ROOT / "outputs/out_240226_amie_cnn",
    "E4B": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/E4B_cnn",
    "TEM": ROOT / "outputs/out_240226_tem_cnn",
    "UBE2I": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/UBE2I_cnn",
}

EVODIFF_SEED_GROW = {
    "AAV": ROOT / "outputs/aav_zero_shot_evodiff_ft_k4_variable_k_kl2_trace_batch64_20260602",
    "LGK": ROOT / "outputs/lgk_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530",
    "GFP": ROOT / "outputs/gfp_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530",
    "Pab1": ROOT / "outputs/pab1_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530",
    "AMIE": ROOT / "outputs/amie_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531",
    "E4B": ROOT / "outputs/e4b_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531",
    "TEM": ROOT / "outputs/tem_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531",
    "UBE2I": ROOT / "outputs/ube2i_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531",
}

EVODIFF_PRETRAIN_TOP16 = {
    "AAV": ROOT / "outputs/aav_zero_shot_evodiff_ft_pretrain_top16_k4_variable_k_kl2_20260601",
    "LGK": ROOT / "outputs/lgk_zero_shot_evodiff_ft_pretrain_top16_k4_variable_k_kl2_20260601",
}

EVODIFF_MIXED_RANK = {
    "AAV": ROOT / "outputs/aav_zero_shot_evodiff_ft_rank_mixed_k4_variable_k_kl2_trace_batch64_20260603",
}

PROSST_FT_ROOTS = {
    8: ROOT / "outputs/prosst_ft_all_landscapes_n8_rank_cluster_20260603",
    128: ROOT / "outputs/prosst_ft_all_landscapes_n128_rank_cluster_20260603",
}


@dataclass(frozen=True)
class Method:
    label: str
    color: str
    marker: str
    root_kind: str
    root: Path | None
    strategy: str | None = None


def build_methods(task: str, budget: int):
    methods = [
        Method("Original CNN", "#16324f", "o", "standard", OG_ROOTS[task], None),
        Method("EvoDiff FT seed_grow", "#c4472d", "s", "standard", EVODIFF_SEED_GROW.get(task), "seed_grow"),
    ]
    if task in EVODIFF_PRETRAIN_TOP16:
        methods.append(Method("EvoDiff pretrain top16 + FT", "#dd8d29", "D", "standard", EVODIFF_PRETRAIN_TOP16[task], "seed_grow"))
    if task in EVODIFF_MIXED_RANK:
        methods.append(Method("EvoDiff FT mixed rank", "#238b45", "P", "standard", EVODIFF_MIXED_RANK[task], "mixed_explore_exploit"))
    methods.append(Method("ProSST FT rank cluster", "#5b3f99", "X", "prosst_all", PROSST_FT_ROOTS[budget], None))
    return methods


def seed_paths(method: Method, task: str, budget: int):
    if method.root is None or not method.root.exists():
        return []
    if method.root_kind == "prosst_all":
        return sorted((method.root / task).glob("seed_*.pkl"))
    if method.strategy is None:
        return sorted((method.root / f"n_samples_{budget}" / task).glob("seed_*.pkl"))
    return sorted((method.root / f"n_samples_{budget}" / method.strategy / task).glob("seed_*.pkl"))


def load_pickle(path: Path):
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        print(f"SKIP unreadable {path}: {exc}")
        return {}


def aggregate_best(method: Method, task: str, budget: int):
    rows = []
    for path in seed_paths(method, task, budget):
        data = load_pickle(path)
        row = {}
        for it, entry in data.items():
            if isinstance(it, int) and isinstance(entry, dict) and "Best score" in entry:
                row[int(it)] = float(entry["Best score"])
        rows.append(row)
    iters = np.arange(1, 11)
    means, sems, counts = [], [], []
    for it in iters:
        vals = np.array([row[it] for row in rows if it in row], dtype=float)
        counts.append(int(len(vals)))
        if len(vals) == 0:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    return iters, np.array(means), np.array(sems), counts


def collect_round_scores(method: Method, task: str, budget: int):
    # Queried candidates are de-duplicated by sequence per round. Retained pool scores are
    # tracked separately as red rugs so the histogram remains the queried distribution.
    queried = defaultdict(dict)
    selected = defaultdict(list)
    paths = seed_paths(method, task, budget)
    for path in paths:
        data = load_pickle(path)
        for it, entry in data.items():
            if not isinstance(it, int) or not isinstance(entry, dict):
                continue
            iter_sequences = entry.get("Iter sequences") or []
            iter_scores = entry.get("Iter scores") or []
            for seq, score in zip(iter_sequences, iter_scores):
                queried[int(it)][str(seq)] = float(score)
            selected_scores = entry.get("Scores") or []
            selected[int(it)].extend(float(score) for score in selected_scores)
    return {
        it: list(seq_to_score.values())
        for it, seq_to_score in queried.items()
    }, dict(selected), len(paths)


def plot_trajectories(output_dir: Path, budgets):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })
    written = []
    summaries = []
    for budget in budgets:
        fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True)
        legend_seen = set()
        for idx, task in enumerate(TASKS):
            ax = axes[idx // 4, idx % 4]
            notes = []
            for method in build_methods(task, budget):
                x, y, e, counts = aggregate_best(method, task, budget)
                valid = ~np.isnan(y)
                if not valid.any():
                    summaries.append(f"SKIP trajectory K={budget} {task} {method.label}: no data")
                    continue
                label = method.label if method.label not in legend_seen else "_nolegend_"
                legend_seen.add(method.label)
                ax.plot(x[valid], y[valid], color=method.color, marker=method.marker, linewidth=2.0, markersize=4.5, label=label)
                ax.fill_between(x[valid], y[valid] - e[valid], y[valid] + e[valid], color=method.color, alpha=0.13, linewidth=0)
                if counts[-1] != 5:
                    notes.append(f"{method.label}: R10 {counts[-1]}/5")
                summaries.append(f"trajectory K={budget} {task} {method.label}: counts={counts}")
            ax.set_title(task)
            ax.set_xlim(1, 10)
            ax.set_xticks(range(1, 11))
            ax.grid(axis="y", alpha=0.22)
            if idx % 4 == 0:
                ax.set_ylabel("Mean max fitness")
            if idx // 4 == 1:
                ax.set_xlabel("Optimization round")
            if notes:
                ax.text(0.02, 0.03, "; ".join(notes), transform=ax.transAxes, fontsize=7.5, color="#555")
        handles, labels = [], []
        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
        fig.suptitle(f"K={budget} Mean-Max Fitness: Original CNN vs EvoDiff/ProSST Fine-Tuning", y=1.055, fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = output_dir / f"ft_methods_vs_original_cnn_k{budget}_mean_max_trajectories.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written, summaries


def plot_histograms(output_dir: Path, budgets, bins):
    hist_dir = output_dir / "fitness_histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)
    written = []
    summaries = []
    for budget in budgets:
        for task in TASKS:
            method_payloads = []
            all_scores = []
            for method in build_methods(task, budget):
                queried, selected, n_seeds = collect_round_scores(method, task, budget)
                if not queried:
                    summaries.append(f"SKIP histogram K={budget} {task} {method.label}: no queried scores")
                    continue
                method_payloads.append((method, queried, selected, n_seeds))
                all_scores.extend(score for scores in queried.values() for score in scores)
                # Include selected pool in axis limits because the red rug can extend beyond queried scores in some runs.
                all_scores.extend(score for scores in selected.values() for score in scores)
            if not method_payloads or not all_scores:
                continue
            score_min = float(np.min(all_scores))
            score_max = float(np.max(all_scores))
            if score_min == score_max:
                score_min -= 0.5
                score_max += 0.5
            pad = 0.03 * (score_max - score_min)
            score_min -= pad
            score_max += pad
            bin_edges = np.linspace(score_min, score_max, bins + 1)
            fig, axes = plt.subplots(10, len(method_payloads), figsize=(4.9 * len(method_payloads), 18), sharex=True, squeeze=False)
            for col, (method, queried, selected, n_seeds) in enumerate(method_payloads):
                axes[0][col].set_title(f"{method.label}\n{n_seeds}/5 seeds", fontsize=10)
                for round_idx in range(1, 11):
                    ax = axes[round_idx - 1][col]
                    scores = queried.get(round_idx, [])
                    ax.hist(scores, bins=bin_edges, color=method.color, alpha=0.82)
                    selected_scores = selected.get(round_idx, [])
                    if selected_scores:
                        y0, y1 = ax.get_ylim()
                        rug_h = max(y1 * 0.08, 1.0)
                        ax.vlines(selected_scores, 0, rug_h, color="#d7191c", alpha=0.22, linewidth=0.55)
                        ax.set_ylim(0, max(y1, rug_h * 1.4))
                    ax.grid(axis="y", alpha=0.22)
                    ax.set_ylabel(f"R{round_idx}\ncount")
                    ax.text(0.98, 0.78, f"n={len(scores)}", ha="right", va="center", transform=ax.transAxes, fontsize=8)
            for ax in axes[-1]:
                ax.set_xlabel("oracle fitness")
            fig.suptitle(
                f"{task} K={budget} queried-candidate fitness distributions\n"
                "Duplicate sequences collapsed per method/round; red rugs show retained pool scores",
                y=0.995,
                fontsize=13,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.975])
            out = hist_dir / f"{task}_k{budget}_ft_methods_vs_original_fitness_histograms.png"
            fig.savefig(out, dpi=180, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
            summaries.append(f"histogram K={budget} {task}: methods={len(method_payloads)}")
    return written, summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original CNN with EvoDiff/ProSST fine-tuning methods.")
    parser.add_argument("--output_dir", default="outputs/rl_vs_og/prosst_ft_methods")
    parser.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument("--skip_histograms", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    summaries = []
    traj_written, traj_summaries = plot_trajectories(output_dir, args.budgets)
    written.extend(traj_written)
    summaries.extend(traj_summaries)
    if not args.skip_histograms:
        hist_written, hist_summaries = plot_histograms(output_dir, args.budgets, args.bins)
        written.extend(hist_written)
        summaries.extend(hist_summaries)

    summary_path = output_dir / "plot_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Written plots:\n")
        for path in written:
            handle.write(f"{path}\n")
        handle.write("\nSummary:\n")
        for line in summaries:
            handle.write(f"{line}\n")
    print(f"Wrote {len(written)} plots")
    print(f"Summary: {summary_path}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
