from pathlib import Path
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "rl_vs_og"
OUT.mkdir(parents=True, exist_ok=True)
TASKS = ["AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I"]
N = 8
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


def files_for(root: Path, task: str, kind: str):
    if kind == "rl":
        return sorted((root / f"n_samples_{N}" / "seed_grow" / task).glob("seed_*.pkl"))
    return sorted((root / f"n_samples_{N}" / task).glob("seed_*.pkl"))


def load_path(path: Path):
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception:
        return {}
    rows = {}
    for it in sorted(k for k in data if isinstance(k, int)):
        row = data[it]
        if "Best score" in row:
            rows[int(it)] = float(row["Best score"])
    return rows


def aggregate(root: Path, task: str, kind: str):
    seed_rows = [load_path(p) for p in files_for(root, task, kind)]
    iters = list(range(1, 11))
    means, sems, counts = [], [], []
    for it in iters:
        vals = np.array([r[it] for r in seed_rows if it in r], dtype=float)
        counts.append(int(len(vals)))
        if len(vals) == 0:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    return np.array(iters), np.array(means), np.array(sems), counts


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})
colors = {"og": "#1f4e79", "rl": "#c4472d"}
labels = {"og": "Original ProSpero CNN", "rl": "Active FT EvoDiff KL=2 seed_grow"}
markers = {"og": "o", "rl": "s"}
fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), sharex=True)
summary = []

for idx, task in enumerate(TASKS):
    ax = axes[idx // 4, idx % 4]
    notes = []
    for kind in ["og", "rl"]:
        x, y, e, counts = aggregate(CONFIGS[task][kind], task, kind)
        valid = ~np.isnan(y)
        if valid.any():
            ax.plot(x[valid], y[valid], color=colors[kind], marker=markers[kind], linewidth=2.2, label=labels[kind])
            ax.fill_between(x[valid], y[valid] - e[valid], y[valid] + e[valid], color=colors[kind], alpha=0.14, linewidth=0)
        final_count = counts[-1]
        if final_count != 5:
            notes.append(f"{kind} iter10 seeds={final_count}/5")
        summary.append(f"{task} {kind}: per-iter seed counts={counts}")
    ax.set_title(task)
    ax.grid(axis="y", alpha=0.22)
    ax.set_xlim(1, 10)
    ax.set_xticks(range(1, 11))
    if idx % 4 == 0:
        ax.set_ylabel("Mean max fitness")
    if idx // 4 == 1:
        ax.set_xlabel("Optimization round")
    if notes:
        ax.text(0.02, 0.03, "; ".join(notes), transform=ax.transAxes, fontsize=8, color="#555")

handles, leglabels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, leglabels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle(
    "K=8 Optimization Trajectories: Active EvoDiff FT vs Original ProSpero CNN\n"
    "Mean max fitness +/- SEM over available completed seeds at each round",
    y=1.045,
    fontsize=15,
    fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
png = OUT / "rl_vs_original_prospero_cnn_k8_mean_max_trajectories.png"
fig.savefig(png, dpi=220, bbox_inches="tight")
print(png)
print("\n".join(summary))
