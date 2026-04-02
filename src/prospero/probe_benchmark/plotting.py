from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `uv add matplotlib`."
    ) from exc


PLOT_METRICS = ("spearman", "r2", "top_decile_recall")


def _task_colors(names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(names)}


def _sorted_config_names(summary: dict) -> list[str]:
    names = set()
    for task_payload in summary["tasks"].values():
        names.update(task_payload["configs"].keys())
    return sorted(names)


def _collect_global_matrix(summary: dict, metric_name: str) -> tuple[list[str], list[str], np.ndarray]:
    task_names = sorted(summary["tasks"])
    config_names = _sorted_config_names(summary)
    matrix = np.full((len(task_names), len(config_names)), np.nan, dtype=float)

    for task_idx, task_name in enumerate(task_names):
        task_payload = summary["tasks"][task_name]
        for config_idx, config_name in enumerate(config_names):
            config_payload = task_payload["configs"].get(config_name)
            if config_payload is None:
                continue
            budget_entries = config_payload["budgets"]
            best_budget = max(budget_entries, key=lambda entry: entry["budget"])
            matrix[task_idx, config_idx] = float(best_budget["metrics"][metric_name]["mean"])

    return task_names, config_names, matrix


def save_plots(summary: dict, output_dir: str | Path, dpi: int = 200) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_task_metric_curves(summary, output_path / "task_metric_curves.png", dpi)
    save_global_metric_curves(summary, output_path / "global_metric_curves.png", dpi)
    for metric_name in PLOT_METRICS:
        save_metric_heatmap(
            summary,
            metric_name,
            output_path / f"heatmap_{metric_name}.png",
            dpi,
        )

    with open(output_path / "plot_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "plots": [
                    "task_metric_curves.png",
                    "global_metric_curves.png",
                    *[f"heatmap_{metric_name}.png" for metric_name in PLOT_METRICS],
                ]
            },
            handle,
            indent=2,
        )


def save_task_metric_curves(summary: dict, out_path: Path, dpi: int) -> None:
    task_names = sorted(summary["tasks"])
    colors = _task_colors(_sorted_config_names(summary))
    fig, axes = plt.subplots(
        len(task_names),
        len(PLOT_METRICS),
        figsize=(5 * len(PLOT_METRICS), 3.5 * len(task_names)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, task_name in enumerate(task_names):
        task_payload = summary["tasks"][task_name]
        for col_idx, metric_name in enumerate(PLOT_METRICS):
            ax = axes[row_idx][col_idx]
            for config_name in sorted(task_payload["configs"]):
                budget_entries = task_payload["configs"][config_name]["budgets"]
                budgets = np.array([entry["budget"] for entry in budget_entries], dtype=float)
                means = np.array(
                    [entry["metrics"][metric_name]["mean"] for entry in budget_entries],
                    dtype=float,
                )
                stds = np.array(
                    [entry["metrics"][metric_name]["std"] for entry in budget_entries],
                    dtype=float,
                )
                ax.plot(
                    budgets,
                    means,
                    label=config_name,
                    color=colors[config_name],
                    linewidth=2,
                )
                ax.fill_between(
                    budgets,
                    means - stds,
                    means + stds,
                    color=colors[config_name],
                    alpha=0.15,
                )
            ax.set_title(f"{task_name}: {metric_name}")
            ax.set_xlabel("Train budget")
            ax.set_ylabel(metric_name)
            ax.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncols=min(3, len(labels)),
        )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_global_metric_curves(summary: dict, out_path: Path, dpi: int) -> None:
    config_names = _sorted_config_names(summary)
    colors = _task_colors(config_names)
    fig, axes = plt.subplots(
        1,
        len(PLOT_METRICS),
        figsize=(5 * len(PLOT_METRICS), 4.5),
        squeeze=False,
        constrained_layout=True,
    )

    for col_idx, metric_name in enumerate(PLOT_METRICS):
        ax = axes[0][col_idx]
        for config_name in config_names:
            budget_to_values: dict[int, list[float]] = {}
            for task_payload in summary["tasks"].values():
                config_payload = task_payload["configs"].get(config_name)
                if config_payload is None:
                    continue
                for entry in config_payload["budgets"]:
                    budget_to_values.setdefault(entry["budget"], []).append(
                        float(entry["metrics"][metric_name]["mean"])
                    )

            budgets = sorted(budget_to_values)
            means = np.array(
                [np.mean(budget_to_values[budget]) for budget in budgets], dtype=float
            )
            stds = np.array(
                [np.std(budget_to_values[budget]) for budget in budgets], dtype=float
            )
            ax.plot(budgets, means, label=config_name, color=colors[config_name], linewidth=2)
            ax.fill_between(
                budgets,
                means - stds,
                means + stds,
                color=colors[config_name],
                alpha=0.15,
            )

        ax.set_title(f"Cross-task average: {metric_name}")
        ax.set_xlabel("Train budget")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncols=min(3, len(labels)),
        )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_metric_heatmap(summary: dict, metric_name: str, out_path: Path, dpi: int) -> None:
    task_names, config_names, matrix = _collect_global_matrix(summary, metric_name)

    fig, ax = plt.subplots(figsize=(1.8 * len(config_names) + 2, 0.5 * len(task_names) + 2))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(config_names)))
    ax.set_xticklabels(config_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(task_names)))
    ax.set_yticklabels(task_names)
    ax.set_title(f"Best-budget {metric_name}")

    for row_idx in range(len(task_names)):
        for col_idx in range(len(config_names)):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                label = "NA"
            else:
                label = f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
