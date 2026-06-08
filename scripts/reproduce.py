#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

TASKS = ["AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I"]
SEEDS = [1, 2, 3, 4, 5]
BUDGETS = [8, 128]


def csv_ints(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the paper-relevant ProSpero/0shotProt runs and plots into "
            "outputs/reproduction/<timestamp>."
        )
    )
    parser.add_argument("--output-root", default="outputs/reproduction")
    parser.add_argument("--timestamp", default=None, help="Override timestamp folder name.")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES value, preferably a GPU UUID.")
    parser.add_argument("--max-workers", type=int, default=5, help="Worker count for vanilla ProSpero variable-K.")
    parser.add_argument("--prosst-batch-size", type=int, default=64)
    parser.add_argument("--prosst-rank-finetune-batch-size", type=int, default=16)
    parser.add_argument("--prosst-grpo-finetune-batch-size", type=int, default=1)
    parser.add_argument("--structure-tokens-dir", default="outputs/prosst_structure_tokens")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--plots-only", action="store_true", default=False)
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["prospero", "rank", "grpo", "noft", "unrestricted", "epistasis"],
        choices=["prospero", "rank", "grpo", "noft", "unrestricted", "epistasis"],
        help="Experiment stages to run before plotting.",
    )
    return parser.parse_args()


class Runner:
    def __init__(self, root: Path, repo_root: Path, env: dict[str, str], dry_run: bool) -> None:
        self.root = root
        self.repo_root = repo_root
        self.env = env
        self.dry_run = dry_run
        self.commands: list[dict[str, object]] = []
        self.logs = root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, cmd: list[str]) -> None:
        log_path = self.logs / f"{len(self.commands):04d}_{name}.log"
        record = {"name": name, "cmd": cmd, "log": str(log_path)}
        self.commands.append(record)
        print("[reproduce]", name)
        print(" ".join(cmd))
        if self.dry_run:
            return
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
            log.flush()
            subprocess.run(cmd, cwd=self.repo_root, env=self.env, check=True, stdout=log, stderr=subprocess.STDOUT)

    def write_manifest(self) -> None:
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"commands": self.commands}, indent=2), encoding="utf-8")
        print(f"[reproduce] manifest: {manifest}")


def seed_done(run_root: Path, task: str, seed: int) -> bool:
    return (run_root / task / f"seed_{seed}.pkl").is_file()


def add_prospero_commands(runner: Runner, args: argparse.Namespace, results: Path) -> None:
    for task in args.tasks:
        out = results / "prospero" / f"{task}_cnn"
        cmd = [
            sys.executable,
            "-m",
            "prospero.runners.run_variable_k",
            str(out),
            "--task",
            task,
            "--surrogate-arch",
            "cnn",
            "--n-samples",
            csv_ints(args.budgets),
            "--seeds",
            csv_ints(args.seeds),
            "--n-iters",
            "10",
            "--max-workers",
            str(args.max_workers),
            "--safe",
        ]
        runner.run(f"prospero_{task}", cmd)


def prosst_common(args: argparse.Namespace, out: Path, task: str, seed: int, budget: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "prospero.runners.run_zero_shot_prosst",
        "--task",
        task,
        "--results_dirpath",
        str(out),
        "--seed",
        str(seed),
        "--n_queries",
        str(budget),
        "--n_iters",
        "10",
        "--batch_size",
        str(args.prosst_batch_size),
        "--mask_budget",
        "4",
        "--mask_strategy",
        "mixed_explore_exploit",
        "--structure_tokens_dir",
        args.structure_tokens_dir,
        "--device",
        "cuda",
        "--debug_generation_trace",
    ]


def add_prosst_stage(runner: Runner, args: argparse.Namespace, results: Path, stage: str) -> None:
    for budget in args.budgets:
        out = results / f"0shotprot_prosst_{stage}_n{budget}"
        for task in args.tasks:
            for seed in args.seeds:
                if args.skip_existing and seed_done(out, task, seed):
                    continue
                cmd = prosst_common(args, out, task, seed, budget)
                if stage in {"rank", "grpo", "unrestricted"}:
                    cmd.extend([
                        "--finetune_prosst",
                        "--finetune_epochs",
                        "5",
                        "--finetune_lr",
                        "3e-5",
                        "--lambda_kl",
                        "2",
                        "--finetune_replay",
                        "all",
                    ])
                if stage == "rank":
                    cmd.extend(["--finetune_batch_size", str(args.prosst_rank_finetune_batch_size), "--reward_mode", "rank", "--smc_vocab", "cluster"])
                elif stage == "grpo":
                    cmd.extend(["--finetune_batch_size", str(args.prosst_grpo_finetune_batch_size), "--reward_mode", "grpo_advantage", "--smc_vocab", "cluster"])
                elif stage == "noft":
                    cmd.extend(["--smc_vocab", "cluster"])
                elif stage == "unrestricted":
                    cmd.extend(["--finetune_batch_size", str(args.prosst_grpo_finetune_batch_size), "--reward_mode", "grpo_advantage", "--smc_vocab", "full"])
                else:
                    raise ValueError(stage)
                runner.run(f"{stage}_n{budget}_{task}_seed{seed}", cmd)


def add_epistasis(runner: Runner, args: argparse.Namespace, results: Path) -> None:
    out = results / "epistasis"
    cmd = [
        sys.executable,
        "-m",
        "prospero.runners.run_epistasis_additivity_test",
        "--tasks",
        *args.tasks,
        "--output-dir",
        str(out),
    ]
    runner.run("epistasis", cmd)


def add_plots(runner: Runner, args: argparse.Namespace, results: Path, plots: Path) -> None:
    rank_k8 = results / "0shotprot_prosst_rank_n8"
    rank_k128 = results / "0shotprot_prosst_rank_n128"
    runner.run(
        "plot_main_rank",
        [
            sys.executable,
            "-m",
            "prospero.runners.plot_simplified_zero_shotprot_mean_max",
            "--output-dir",
            str(plots / "main_rank"),
            "--prosst-k8-root",
            str(rank_k8),
            "--prosst-k128-root",
            str(rank_k128),
            "--prospero-results-dir",
            str(results / "prospero"),
            "--no-evodiff",
        ],
    )
    for stage in ["rank", "grpo", "noft", "unrestricted"]:
        for budget in args.budgets:
            run_dir = results / f"0shotprot_prosst_{stage}_n{budget}"
            if args.dry_run or run_dir.exists():
                runner.run(
                    f"plot_hist_{stage}_n{budget}",
                    [
                        sys.executable,
                        "-m",
                        "prospero.runners.plot_zero_shot_round_fitness_histograms",
                        "--run_dir",
                        str(run_dir),
                        "--method_label",
                        f"0shotProt ProSST {stage} K={budget}",
                        "--output_dir",
                        str(plots / "round_histograms" / stage / f"k{budget}"),
                        "--tasks",
                        *args.tasks,
                    ],
                )
    epi_json = results / "epistasis" / "epistasis_additivity_all_tasks.json"
    if args.dry_run or epi_json.exists():
        runner.run(
            "plot_epistasis",
            [
                sys.executable,
                "-m",
                "prospero.runners.plot_epistasis_additivity_styled",
                "--source",
                str(epi_json),
                "--output-dir",
                str(plots / "epistasis"),
            ],
        )
    if 128 in args.budgets:
        runner.run(
            "plot_vocab_ablation_k128",
            [
                sys.executable,
                "-m",
                "prospero.runners.plot_prosst_vocab_ablation_k128",
                "--output-dir",
                str(plots / "vocab_ablation_k128"),
                "--restricted-root",
                str(results / "0shotprot_prosst_grpo_n128"),
                "--unrestricted-root",
                str(results / "0shotprot_prosst_unrestricted_n128"),
                "--tasks",
                *args.tasks,
            ],
        )


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / args.output_root / timestamp
    results = root / "results"
    plots = root / "plots"
    root.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    runner = Runner(root=root, repo_root=repo_root, env=env, dry_run=args.dry_run)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"[reproduce] root: {root}")

    if not args.plots_only:
        if "prospero" in args.stages:
            add_prospero_commands(runner, args, results)
        for stage in ["rank", "grpo", "noft", "unrestricted"]:
            if stage in args.stages:
                add_prosst_stage(runner, args, results, stage)
        if "epistasis" in args.stages:
            add_epistasis(runner, args, results)

    if not args.no_plots:
        add_plots(runner, args, results, plots)

    runner.write_manifest()


if __name__ == "__main__":
    main()
