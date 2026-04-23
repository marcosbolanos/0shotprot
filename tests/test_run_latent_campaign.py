from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

from prospero.runners import run_latent_campaign


def test_build_fallbacks_deduped_and_descending() -> None:
    pairs = run_latent_campaign._build_fallbacks(batch_size=256, combo_chunk_size=4)
    assert pairs[0] == (256, 4)
    assert pairs[-1] == (32, 1)
    assert len(pairs) == len(set(pairs))


def test_campaign_retries_after_oom_and_succeeds(tmp_path: Path, monkeypatch) -> None:
    out_root = tmp_path / "outputs_latent_campaign"
    calls: list[tuple[int, int]] = []
    seen_phases: list[str] = []

    def _fake_run(args: SimpleNamespace) -> Path:
        calls.append((int(args.batch_size), int(args.combo_chunk_size)))
        seen_phases.append(str(args.phase))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        run_payload = {
            "best_oracle": 0.5,
            "oracle_top10_mean": 0.4,
            "surrogate_oracle_corr": 0.2,
            "num_generated": 12,
            "num_unique": 10,
            "mean_mutation_count": 3.0,
            "best_sequence": "ACDE",
            "mutated_positions": [1, 2],
            "timings_seconds": {"total_runtime_seconds": 1.25},
        }
        out_dir = out_root / "LGK"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"seed_{args.seed}_latent_masked_single_iter.json"
        out_path.write_text(json.dumps(run_payload), encoding="utf-8")
        return out_path

    monkeypatch.setattr(run_latent_campaign, "run_latent_masked_single_iter", _fake_run)
    monkeypatch.setattr(
        run_latent_campaign, "get_parser", lambda: _parser_for_test(out_root)
    )
    monkeypatch.setattr(sys, "argv", ["run_latent_campaign.py"])

    run_latent_campaign.main()

    assert calls[:2] == [(256, 4), (128, 2)]
    assert seen_phases == ["phase0_baseline", "phase0_baseline"]

    rows_path = out_root / "LGK" / "phase0_baseline" / "runs.jsonl"
    summary_path = out_root / "LGK" / "phase0_baseline" / "campaign_summary.json"
    assert rows_path.exists()
    assert summary_path.exists()

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["attempt"] == 2
    assert rows[0]["used_batch_size"] == 128
    assert rows[0]["used_combo_chunk_size"] == 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["num_ok"] == 1
    assert summary["num_error"] == 0
    assert summary["best_oracle_max"] == 0.5


def _parser_for_test(out_root: Path):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dirpath", type=str, default=str(out_root))
    parser.add_argument("--task", type=str, default="LGK")
    parser.add_argument("--phase", type=str, default="phase0_baseline")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=1)
    parser.add_argument("--surrogate-arch", type=str, default="interplm_mean_pool_ridge")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--top-features", type=int, default=3)
    parser.add_argument("--steering-layer", type=int, default=2)
    parser.add_argument(
        "--steering-direction-mode",
        type=str,
        default="signed",
        choices=["signed", "both", "positive", "negative"],
    )
    parser.add_argument("--steering-scalars", type=float, nargs="+", default=[0.2, 0.7, 1.2, 1.8])
    parser.add_argument("--combo-chunk-size", type=int, default=4)
    parser.add_argument("--mutation-penalty-lambda", type=float, default=0.0)
    parser.add_argument("--max-mutations", type=int, default=None)
    parser.add_argument("--max-masked-mean-ppl", type=float, default=None)
    parser.add_argument("--max-masked-token-ppl", type=float, default=None)
    parser.add_argument("--min-corruptions", type=int, default=3)
    parser.add_argument("--max-corruptions", type=int, default=10)
    parser.add_argument("--n-checks-multiplier", type=int, default=16)
    parser.add_argument("--kappa-scan", type=float, default=1.0)
    parser.add_argument("--kappa-guidance", type=float, default=0.1)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--proxy-batch-size", type=int, default=256)
    parser.add_argument("--num-model-max-epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs-per-valid", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-fit-intercept", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--esm-model-name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--esm-max-length", type=int, default=None)
    parser.add_argument("--interplm-layer", type=int, default=2)
    parser.add_argument("--interplm-repo-id", type=str, default="Elana/InterPLM-esm2-8m")
    parser.add_argument("--interplm-normalized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sae-token-chunk-size", type=int, default=1024)
    parser.add_argument("--alphabet", type=str, default="CHARGE")
    parser.add_argument("--disable-esm-cache", action="store_true", default=False)
    parser.add_argument("--full-deterministic", action="store_true", default=False)
    return parser
