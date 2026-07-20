from pathlib import Path

from prospero.reproduction.commands import alignment_commands, zero_shot_commands
from prospero.reproduction.types import AlignmentStage, ReproductionContext, ZeroShotStage


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


def test_evodiff_stage_uses_final_protocol_runner(tmp_path):
    stage = ZeroShotStage(
        name="evodiff_finetuned",
        tasks=("AAV",),
        plm="evodiff",
        budgets=(8,),
        seeds=(1,),
        finetune=True,
    )

    [(name, command)] = zero_shot_commands(stage, context(tmp_path))

    assert name == "evodiff_finetuned_n8_AAV_seed1"
    assert "prospero.runners.run_zero_shot_evodiff" in command
    assert "--finetune_evodiff" in command
    assert "--structure_tokens_dir" not in command
    assert command[command.index("--finetune_lr") + 1] == "3e-05"


def test_alignment_stage_is_self_contained(tmp_path):
    stage = AlignmentStage(tasks=("AAV",), max_sequences=128)

    [(name, command)] = alignment_commands(stage, context(tmp_path))

    assert name == "plm_mms_pll_alignment"
    assert "prospero.runners.run_plm_full_pll_alignment" in command
    assert command[command.index("--max_sequences") + 1] == "128"
    assert command[command.index("--plms") + 1 : command.index("--max_sequences")] == [
        "evodiff",
        "esm",
        "prosst",
    ]
