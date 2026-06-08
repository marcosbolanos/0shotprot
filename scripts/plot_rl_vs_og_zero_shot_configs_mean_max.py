from pathlib import Path
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "rl_vs_og"
OUT.mkdir(parents=True, exist_ok=True)
N = 128
TASKS = ["AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I"]

ACTIVE_ROOTS = {
    ROOT / "outputs/aav_zero_shot_evodiff_ft_bottomq_mixed_k4_n128_kl2_neg05_trace_batch64_20260603",
    ROOT / "outputs/aav_zero_shot_evodiff_ft_bottomq_mixed_k4_n128_kl2_neg10_trace_batch64_20260603",
    ROOT / "outputs/aav_zero_shot_evodiff_ft_grpo_mixed_k4_n128_kl2_neg05_trace_batch64_20260603",
    ROOT / "outputs/aav_zero_shot_evodiff_ft_grpo_mixed_k4_n128_kl2_neg10_trace_batch64_20260603",
    ROOT / "outputs/aav_zero_shot_evodiff_ft_rank_mixed_fullsmc_k4_n128_kl2_trace_batch64_20260603",
    ROOT / "outputs/aav_zero_shot_evodiff_ft_rank_mixed_no_rollout_cluster_k4_n128_kl2_trace_batch64_20260603",
    ROOT / "outputs/aav_zero_shot_evodiff_ft_rank_mixed_no_rollout_full_k4_n128_kl2_trace_batch64_20260603",
}

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

# Each entry: label, root, strategy subdir under n_samples_128, marker, color.
COMMON_ZS = {
    "AAV": [
        ("no FT seed_grow", ROOT / "outputs/aav_zero_shot_fixed_mask_k4_trace_batch64_20260602", "seed_grow", "^", "#8c6bb1"),
        ("FT seed_grow KL=2", ROOT / "outputs/aav_zero_shot_evodiff_ft_k4_variable_k_20260529", "seed_grow", "s", "#c4472d"),
        ("FT seed_grow KL=2 traced", ROOT / "outputs/aav_zero_shot_evodiff_ft_k4_variable_k_kl2_trace_batch64_20260602", "seed_grow", "P", "#ef8a62"),
        ("FT mixed rank KL=2", ROOT / "outputs/aav_zero_shot_evodiff_ft_rank_mixed_k4_variable_k_kl2_trace_batch64_20260603", "mixed_explore_exploit", "D", "#238b45"),
        ("FT GRPO seed_grow KL=2", ROOT / "outputs/aav_zero_shot_evodiff_ft_grpo_k4_variable_k_kl2_trace_batch64_20260603", "seed_grow", "v", "#2b8cbe"),
        ("FT GRPO mixed KL=2", ROOT / "outputs/aav_zero_shot_evodiff_ft_grpo_mixed_k4_variable_k_kl2_trace_batch64_20260603", "mixed_explore_exploit", "X", "#08589e"),
    ],
    "LGK": [
        ("FT seed_grow KL=2", ROOT / "outputs/lgk_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530", "seed_grow", "s", "#c4472d"),
        ("FT seed_grow KL=0.1", ROOT / "outputs/lgk_zero_shot_evodiff_ft_k4_variable_k_kl0p1_20260530", "seed_grow", "o", "#ef8a62"),
        ("pretrain top16 + FT", ROOT / "outputs/lgk_zero_shot_evodiff_ft_pretrain_top16_k4_variable_k_kl2_20260601", "seed_grow", "D", "#238b45"),
    ],
    "GFP": [
        ("FT seed_grow KL=2", ROOT / "outputs/gfp_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530", "seed_grow", "s", "#c4472d"),
    ],
    "Pab1": [
        ("FT seed_grow KL=2", ROOT / "outputs/pab1_zero_shot_evodiff_ft_k4_variable_k_kl2_20260530", "seed_grow", "s", "#c4472d"),
    ],
    "AMIE": [
        ("FT seed_grow KL=2", ROOT / "outputs/amie_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531", "seed_grow", "s", "#c4472d"),
    ],
    "E4B": [
        ("FT seed_grow KL=2", ROOT / "outputs/e4b_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531", "seed_grow", "s", "#c4472d"),
    ],
    "TEM": [
        ("FT seed_grow KL=2", ROOT / "outputs/tem_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531", "seed_grow", "s", "#c4472d"),
    ],
    "UBE2I": [
        ("FT seed_grow KL=2", ROOT / "outputs/ube2i_zero_shot_evodiff_ft_k4_variable_k_kl2_20260531", "seed_grow", "s", "#c4472d"),
    ],
}


def seed_paths(root: Path, task: str, strategy: str | None):
    if root in ACTIVE_ROOTS:
        return []
    if strategy is None:
        return sorted((root / f"n_samples_{N}" / task).glob("seed_*.pkl"))
    return sorted((root / f"n_samples_{N}" / strategy / task).glob("seed_*.pkl"))


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


def aggregate(root: Path, task: str, strategy: str | None):
    paths = seed_paths(root, task, strategy)
    rows = [load_path(p) for p in paths]
    iters = np.arange(1, 11)
    means, sems, counts = [], [], []
    for it in iters:
        vals = np.array([r[it] for r in rows if it in r], dtype=float)
        counts.append(int(len(vals)))
        if len(vals) == 0:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    return iters, np.array(means), np.array(sems), counts


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})
fig, axes = plt.subplots(2, 4, figsize=(21, 10), sharex=True)
summary = []
legend_seen = {}

for idx, task in enumerate(TASKS):
    ax = axes[idx // 4, idx % 4]
    series = [("Original CNN", OG_ROOTS[task], None, "o", "#1f4e79")] + COMMON_ZS.get(task, [])
    notes = []
    for label, root, strategy, marker, color in series:
        x, y, e, counts = aggregate(root, task, strategy)
        valid = ~np.isnan(y)
        if not valid.any():
            summary.append(f"SKIP {task} {label}: no usable completed data at K={N}")
            continue
        display_label = label if label not in legend_seen else "_nolegend_"
        legend_seen[label] = True
        ax.plot(x[valid], y[valid], color=color, marker=marker, linewidth=2.0, markersize=4.5, label=display_label)
        ax.fill_between(x[valid], y[valid] - e[valid], y[valid] + e[valid], color=color, alpha=0.12, linewidth=0)
        if counts[-1] != 5:
            notes.append(f"{label}: iter10 {counts[-1]}/5")
        summary.append(f"{task} {label}: root={root.relative_to(ROOT)} strategy={strategy} counts={counts}")
    ax.set_title(task)
    ax.grid(axis="y", alpha=0.22)
    ax.set_xlim(1, 10)
    ax.set_xticks(range(1, 11))
    if idx % 4 == 0:
        ax.set_ylabel("Mean max fitness")
    if idx // 4 == 1:
        ax.set_xlabel("Optimization round")
    if notes:
        ax.text(0.02, 0.03, "; ".join(notes), transform=ax.transAxes, fontsize=7.5, color="#555")

handles, labels = [], []
for ax in axes.flat:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h); labels.extend(l)
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.suptitle(
    "K=128 Mean-Max Fitness Trajectories: Original CNN vs Completed Zero-Shot Configs\n"
    "Each zero-shot strategy is a separate curve; active AAV ablations excluded",
    y=1.06,
    fontsize=15,
    fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
png = OUT / "rl_vs_original_cnn_zero_shot_configs_k128_mean_max_trajectories.png"
fig.savefig(png, dpi=220, bbox_inches="tight")
print(png)
print("\n".join(summary))
