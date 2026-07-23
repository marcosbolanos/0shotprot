#!/usr/bin/env python
from __future__ import annotations

from prospero.reproduction import (
    DecodingVocabulary,
    EpistasisStage,
    OnlineAdaptationConfig,
    PlmOptimizationStage,
    ProSperoStage,
    ProteinLanguageModel,
    ReproductionRecipe,
    ScoringBenchmarkStage,
    run_reproduction,
)
from prospero.reproduction.cli import parse_runtime_options


TASKS = ("AAV", "LGK", "GFP", "Pab1", "AMIE", "E4B", "TEM", "UBE2I")
SEEDS = (1, 2, 3, 4, 5)
QUERY_BUDGETS = (8, 128)
STRUCTURE_TOKENS = "assets/prosst_structure_tokens"

ONLINE_ADAPTATION = OnlineAdaptationConfig(
    epochs=5,
    learning_rate=3e-5,
    kl_coefficient=2.0,
    batch_size=1,
)

RECIPE = ReproductionRecipe(
    stages=(
        ProSperoStage(
            name="prospero_cnn",
            tasks=TASKS,
            query_budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            rounds=10,
            max_workers=5,
        ),
        PlmOptimizationStage(
            name="prosst_online_adaptation",
            tasks=TASKS,
            model=ProteinLanguageModel.PROSST,
            plot_label="0shotProt (w/ ProSST)",
            query_budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            adaptation=ONLINE_ADAPTATION,
            decoding_vocabulary=DecodingVocabulary.RESTRICTED,
            structure_tokens_directory=STRUCTURE_TOKENS,
        ),
        PlmOptimizationStage(
            name="evodiff_online_adaptation",
            tasks=TASKS,
            model=ProteinLanguageModel.EVODIFF,
            plot_label="0shotProt (w/ EvoDiff)",
            query_budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            adaptation=ONLINE_ADAPTATION,
            decoding_vocabulary=DecodingVocabulary.RESTRICTED,
        ),
        PlmOptimizationStage(
            name="prosst_unrestricted_vocabulary",
            tasks=TASKS,
            model=ProteinLanguageModel.PROSST,
            plot_label="0shotProt (unrestricted vocabulary)",
            query_budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            adaptation=ONLINE_ADAPTATION,
            decoding_vocabulary=DecodingVocabulary.UNRESTRICTED,
            structure_tokens_directory=STRUCTURE_TOKENS,
        ),
        PlmOptimizationStage(
            name="prosst_without_adaptation",
            tasks=TASKS,
            model=ProteinLanguageModel.PROSST,
            plot_label="0shotProt (no fine-tuning)",
            query_budgets=QUERY_BUDGETS,
            seeds=SEEDS,
            adaptation=None,
            decoding_vocabulary=DecodingVocabulary.RESTRICTED,
            structure_tokens_directory=STRUCTURE_TOKENS,
        ),
        ScoringBenchmarkStage(
            tasks=TASKS,
            models=("evodiff", "esm", "prosst"),
            max_sequences=128,
            chunk_size=4,
            seed=142857,
            structure_tokens_directory=STRUCTURE_TOKENS,
        ),
        EpistasisStage(
            tasks=TASKS,
            seed=1,
            samples_per_pair_type=250,
            oracle_batch_size=128,
        ),
    )
)


if __name__ == "__main__":
    options = parse_runtime_options(tuple(stage.name for stage in RECIPE.stages))
    run_reproduction(RECIPE, options)
