from pathlib import Path
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "rl_vs_og"
OUT.mkdir(parents=True, exist_ok=True)
BUDGETS = [8, 16, 32, 64, 128]

TASKS = ["AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I"]
CONFIGS = {
    "AAV": {
        "og": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/AAV_cnn",
        "rl": ROOT / "outputs/aav_zero_shot_evodiff_ft_k4_variable_k_20260529",
    },
    "LGK": {
        "og": ROOT / "outputs/out_240226_lgk_cnn",
        "rl": ROOT / "outputs/lgk_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530",
    },
    "GFP": {
        "og": ROOT / "outputs/out_240226_gfp_cnn",
        "rl": ROOT / "outputs/gfp_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530",
    },
    "Pab1": {
        "og": ROOT / "outputs/out_240226_pab1_cnn",
        "rl": ROOT / "outputs/pab1_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530",
    },
    "AMIE": {
        "og": ROOT / "outputs/out_240226_amie_cnn",
        "rl": ROOT / "outputs/amie_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531",
    },
    "E4B": {
        "og": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/E4B_cnn",
        "rl": ROOT / "outputs/e4b_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531",
    },
    "TEM": {
        "og": ROOT / "outputs/out_240226_tem_cnn",
        "rl": ROOT / "outputs/tem_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531",
    },
    "UBE2I": {
        "og": ROOT / "outputs/variable_k_cnn_excl_set_noa6000_20260504_175400/UBE2I_cnn",
        "rl": ROOT / "outputs/ube2i_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531",
    },
}


def load_seed_metrics(path: Path):
    with path.open("rb") as handle:
        data = pickle.load(handle)
    keys = sorted(k for k in data if isinstance(k, int))
    if not keys:
        return None
    iteration = keys[-1]
    row = data[iteration]
    return {
        "iter": iteration,
        "perf": float(row["Performance"]),
        "best": float(row["Best score"]),
    }


def files_for(root: Path, task: str, budget: int, kind: str):
    if kind == "rl":
        return sorted((root / f"n_samples_{budget}" / "seed_grow" / task).glob("seed_*.pkl"))
    return sorted((root / f"n_samples_{budget}" / task).glob("seed_*.pkl"))


def aggregate(root: Path, task: str, kind: str):
    out = {"perf_mean": [], "perf_sem": [], "best_mean": [], "best_sem": [], "n": []}
    for budget in BUDGETS:
        vals = []
        for path in files_for(root, task, budget, kind):
            try:
                metrics = load_seed_metrics(path)
            except Exception:
                metrics = None
            if metrics and metrics["iter"] >= 10:
                vals.append(metrics)
        out["n"].append(len(vals))
        for metric in ["perf", "best"]:
            arr = np.array([v[metric] for v in vals], dtype=float)
            if len(arr) == 0:
                mean = np.nan
                sem = np.nan
            else:
                mean = float(np.mean(arr))
                sem = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
            out[f"{metric}_mean"].append(mean)
            out[f"{metric}_sem"].append(sem)
    return out


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})

fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), sharex=True)
colors = {"og": "#1f4e79", "rl": "#c4472d"}
labels = {"og": "Original ProSpero CNN", "rl": "Active FT EvoDiff KL=2 seed_grow"}
markers = {"og": "o", "rl": "s"}
summary_lines = []

for idx, task in enumerate(TASKS):
    row_block = 0 if idx < 4 else 1
    col = idx % 4
    ax = axes[row_block, col]
    cfg = CONFIGS[task]
    data = {kind: aggregate(path, task, kind) for kind, path in cfg.items()}
    for kind in ["og", "rl"]:
        ax.errorbar(
            BUDGETS,
            data[kind]["best_mean"],
            yerr=data[kind]["best_sem"],
            label=labels[kind],
            color=colors[kind],
            marker=markers[kind],
            linewidth=2.2,
            capsize=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.grid(axis="y", alpha=0.22)
    ax.set_title(task)
    if col == 0:
        ax.set_ylabel("Mean max fitness")
    if row_block == 1:
        ax.set_xlabel("Sequences per round")
    notes = []
    for kind in ["og", "rl"]:
        bad = [f"{b}:{n}" for b, n in zip(BUDGETS, data[kind]["n"]) if n != 5]
        if bad:
            notes.append(f"{kind} seeds " + ", ".join(bad))
    if notes:
        ax.text(0.02, 0.03, "; ".join(notes), transform=ax.transAxes, fontsize=8, color="#555")
    summary_lines.append(f"{task}: OG n={data['og']['n']}; RL n={data['rl']['n']}")

handles, leglabels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, leglabels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle(
    "Active EvoDiff Fine-Tuning vs Original ProSpero CNN\n"
    "KL=2, seed_grow masking; mean max fitness +/- SEM over completed seeds",
    y=1.045,
    fontsize=15,
    fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

png = OUT / "rl_vs_original_prospero_cnn_variable_k_mean_max.png"
fig.savefig(png, dpi=220, bbox_inches="tight")
print(png)
print("\n".join(summary_lines))
