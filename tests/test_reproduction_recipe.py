from prospero.reproduction.commands import (
    optimization_commands,
    plot_commands,
    scoring_benchmark_commands,
)
from prospero.reproduction.types import (
    DecodingVocabulary,
    OnlineAdaptationConfig,
    PlmOptimizationStage,
    ProSperoStage,
    ProteinLanguageModel,
    ReproductionContext,
    ScoringBenchmarkStage,
)
from prospero.runners.run_prosst_optimization import (
    build_configs as build_prosst_configs,
    get_parser as get_prosst_parser,
)


def context(tmp_path):
    root = tmp_path / "reproduction"
    return ReproductionContext(
        repo_root=tmp_path,
        root=root,
        results=root / "results",
        plots=root / "plots",
        logs=root / "logs",
        env={},
        dry_run=True,
        skip_existing=False,
    )


def test_evodiff_stage_uses_shared_optimization_protocol(tmp_path):
    stage = PlmOptimizationStage(
        name="evodiff_online_adaptation",
        tasks=("AAV",),
        model=ProteinLanguageModel.EVODIFF,
        query_budgets=(8,),
        seeds=(1,),
        adaptation=OnlineAdaptationConfig(),
    )

    [(name, command)] = optimization_commands(stage, context(tmp_path))

    assert name == "evodiff_online_adaptation_k8_AAV_seed1"
    assert "prospero.runners.run_evodiff_optimization" in command
    assert "--online-adaptation" in command
    assert "--structure-tokens-directory" not in command
    assert command[command.index("--adaptation-learning-rate") + 1] == "3e-05"


def test_scoring_benchmark_stage_is_self_contained(tmp_path):
    stage = ScoringBenchmarkStage(tasks=("AAV",), max_sequences=128)

    [(name, command)] = scoring_benchmark_commands(stage, context(tmp_path))

    assert name == "plm_scoring_benchmark"
    assert "prospero.runners.run_plm_scoring_benchmark" in command
    assert command[command.index("--max-sequences") + 1] == "128"
    assert command[
        command.index("--models") + 1 : command.index("--max-sequences")
    ] == [
        "evodiff",
        "esm",
        "prosst",
    ]


def test_combined_plot_requires_both_query_budgets(tmp_path):
    stages = [
        ProSperoStage(tasks=("AAV",), query_budgets=(8,)),
        PlmOptimizationStage(
            name="prosst_online_adaptation",
            tasks=("AAV",),
            query_budgets=(8,),
            seeds=(1,),
        ),
    ]

    commands = plot_commands(stages, context(tmp_path))

    assert "plot_main_optimization" not in {name for name, _ in commands}


def test_no_adaptation_stage_cannot_load_adapted_model(tmp_path):
    arguments = get_prosst_parser().parse_args(
        [
            "--results-directory",
            str(tmp_path),
            "--task",
            "AAV",
            "--decoding-vocabulary",
            DecodingVocabulary.RESTRICTED.value,
        ]
    )

    _, model_config = build_prosst_configs(arguments)

    assert model_config.adaptation is None
