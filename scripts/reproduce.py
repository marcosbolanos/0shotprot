#!/usr/bin/env python
from __future__ import annotations

from prospero.reproduction import (
    AlignmentStage,
    EpistasisStage,
    ProSperoStage,
    ReproductionRecipe,
    ZeroShotStage,
    run_reproduction,
)
from prospero.reproduction.cli import parse_runtime_options

# Paper landscapes and query budgets.
TASKS = ("AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I")
ALIGNMENT_TASKS = ("AAV", "AMIE", "E4B", "GFP", "LGK", "Pab1", "TEM", "UBE2I")
SEEDS = (1, 2, 3, 4, 5)
QUERY_BUDGETS = (8, 128)

# Shared 0shotProt settings.
PROSST_MODEL_INPUTS = {
    "structure_tokens_dir": "outputs/prosst_structure_tokens",
    "mask_budget": 4,
    "batch_size": 64,
}

PROSST_FINETUNE = {
    "finetune": True,
    "finetune_epochs": 5,
    "finetune_lr": 3e-5,
    "lambda_kl": 2.0,
}

EVODIFF_MODEL_INPUTS = {
    "plm": "evodiff",
    "mask_budget": 4,
    "batch_size": 64,
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
            name="prosst_finetuned",
            tasks=TASKS,
            plm="prosst",
            plot_label="0shotProt (w/ ProSST)",
            budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            decoding_vocab="restricted",
            finetune_batch_size=1,
            **PROSST_MODEL_INPUTS,
            **PROSST_FINETUNE,
        ),
        ZeroShotStage(
            name="evodiff_finetuned",
            tasks=TASKS,
            plot_label="0shotProt (w/ EvoDiff)",
            budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            decoding_vocab="restricted",
            finetune=True,
            finetune_epochs=5,
            finetune_lr=3e-5,
            lambda_kl=2.0,
            finetune_batch_size=1,
            **EVODIFF_MODEL_INPUTS,
        ),
        ZeroShotStage(
            name="prosst_finetuned_unrestricted",
            tasks=TASKS,
            plm="prosst",
            plot_label="0shotProt (unrestricted vocabulary)",
            budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            decoding_vocab="unrestricted",
            finetune_batch_size=1,
            **PROSST_MODEL_INPUTS,
            **PROSST_FINETUNE,
        ),
        ZeroShotStage(
            name="no_finetune",
            tasks=TASKS,
            plm="prosst",
            plot_label="0shotProt (no fine-tuning)",
            budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            decoding_vocab="restricted",
            finetune=False,
            **PROSST_MODEL_INPUTS,
        ),
        AlignmentStage(
            name="plm_mms_pll_alignment",
            tasks=ALIGNMENT_TASKS,
            plms=("evodiff", "esm", "prosst"),
            max_sequences=128,
            chunk_size=4,
            seed=142857,
            structure_tokens_dir=PROSST_MODEL_INPUTS["structure_tokens_dir"],
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
