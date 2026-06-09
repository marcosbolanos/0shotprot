#!/usr/bin/env python
from __future__ import annotations

from prospero.reproduction import (
    EpistasisStage,
    ProSperoStage,
    ReproductionRecipe,
    ZeroShotStage,
    run_reproduction,
)
from prospero.reproduction.cli import parse_runtime_options

# Paper landscapes and query budgets.
TASKS = ("AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I")
SEEDS = (1, 2, 3, 4, 5)
QUERY_BUDGETS = (8, 128)

# Shared 0shotProt settings.
PROSST_MODEL_INPUTS = {
    "structure_tokens_dir": "outputs/prosst_structure_tokens",
    "mask_strategy": "mixed_explore_exploit",
    "mask_budget": 4,
    "batch_size": 64,
}

PROSST_FINETUNE = {
    "finetune": True,
    "finetune_epochs": 5,
    "finetune_lr": 3e-5,
    "lambda_kl": 2.0,
}

RECIPE = ReproductionRecipe(
    stages=(
        ProSperoStage(
            name="prospero_cnn_variable_k",
            tasks=TASKS,
            budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            surrogate_arch="cnn",
            n_iters=10,
            max_workers=5,
        ),
        ZeroShotStage(
            name="grpo_cluster",
            tasks=TASKS,
            budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            loss="grpo_advantage",
            smc_vocab="cluster",
            finetune_batch_size=1,
            **PROSST_MODEL_INPUTS,
            **PROSST_FINETUNE,
        ),
        ZeroShotStage(
            name="grpo_unrestricted_vocab",
            tasks=TASKS,
            budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            loss="grpo_advantage",
            smc_vocab="full",
            finetune_batch_size=1,
            **PROSST_MODEL_INPUTS,
            **PROSST_FINETUNE,
        ),
        EpistasisStage(
            name="epistasis_additivity",
            tasks=TASKS,
            seed=1,
            samples_per_pair_type=250,
            oracle_batch_size=128,
        ),
    ),
    plot_after_runs=True,
)


if __name__ == "__main__":
    options = parse_runtime_options(tuple(stage.name for stage in RECIPE.stages))
    run_reproduction(RECIPE, options)
