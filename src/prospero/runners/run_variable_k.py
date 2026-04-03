"""Run the variable-n_queries experiments from Python instead of bash."""

import argparse
import concurrent.futures
from datetime import datetime
import os
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence, TextIO

from tqdm import tqdm


class TeeStream:
    """Duplicate writes so console output is mirrored in the log file."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(
            getattr(stream, "isatty", lambda: False)() for stream in self._streams
        )


@dataclass(frozen=True)
class SharedESMComponents:
    tokenizer: object
    esm: object
    esm_forward_lock: object | None = None
    esm_batch_worker: object | None = None
    esm_in_memory_cache_pool: object | None = None


def parse_int_list(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value}") from error


def _run_command(process_args: Sequence[str], env: Optional[dict[str, str]]) -> None:
    run_env = dict(env) if env is not None else os.environ.copy()
    # Encourage child Python processes to flush output promptly.
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    run_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    print(f"[driver] launching: {' '.join(process_args)}", flush=True)

    with subprocess.Popen(
        process_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=run_env,
        text=True,
        bufsize=1,
    ) as process:
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process_args)


def run_seed(process_args: Sequence[str], env: dict[str, str]) -> None:
    _run_command(process_args, env)


def prepare_shared_esm_components(args: argparse.Namespace) -> SharedESMComponents:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from prospero.surrogate import SharedESMBatchWorker, SharedInMemoryESMCachePool

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[shared-esm] loading tokenizer from local cache", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.esm_model_name,
            local_files_only=True,
        )
    except Exception:
        print("[shared-esm] local tokenizer cache miss, falling back to hub", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.esm_model_name)

    print("[shared-esm] loading model from local cache", flush=True)
    try:
        esm = AutoModel.from_pretrained(
            args.esm_model_name,
            local_files_only=True,
        ).to(device)
    except Exception:
        print("[shared-esm] local model cache miss, falling back to hub", flush=True)
        esm = AutoModel.from_pretrained(args.esm_model_name).to(device)

    print(f"[shared-esm] shared ESM model loaded on {device}", flush=True)
    esm.eval()
    for param in esm.parameters():
        param.requires_grad = False
    esm_batch_worker = SharedESMBatchWorker(
        tokenizer=tokenizer,
        esm=esm,
        device=device,
        max_batch_sequences=512,
        max_wait_ms=4.0,
    )
    esm_in_memory_cache_pool = SharedInMemoryESMCachePool(storage_dtype=torch.float16)
    return SharedESMComponents(
        tokenizer=tokenizer,
        esm=esm,
        esm_forward_lock=None,
        esm_batch_worker=esm_batch_worker,
        esm_in_memory_cache_pool=esm_in_memory_cache_pool,
    )


def _build_protein_args(
    runner_args: argparse.Namespace,
    batch_dir: Path,
    n_iters: int,
    n_queries: int,
    seed: int,
) -> argparse.Namespace:
    from prospero.runners.run_protein import get_parser as get_protein_parser

    protein_args = get_protein_parser().parse_args([])
    protein_args.task = runner_args.task
    protein_args.results_dirpath = str(batch_dir)
    protein_args.n_iters = n_iters
    protein_args.min_corruptions = runner_args.min_corruptions
    protein_args.max_corruptions = runner_args.max_corruptions
    protein_args.surrogate_arch = runner_args.surrogate_arch
    protein_args.full_deterministic = True
    protein_args.n_queries = n_queries
    protein_args.seed = seed
    protein_args.esm_cnn_projection_dim = runner_args.esm_cnn_projection_dim
    protein_args.esm_cnn_use_layernorm = runner_args.esm_cnn_use_layernorm
    protein_args.esm_cnn_concat_one_hot = runner_args.esm_cnn_concat_one_hot
    protein_args.esm_model_name = runner_args.esm_model_name
    protein_args.esm_max_length = runner_args.esm_max_length
    protein_args.esm_run_cache_base_dir = runner_args.esm_run_cache_base_dir
    protein_args.ridge_alpha = runner_args.ridge_alpha
    protein_args.ridge_fit_intercept = runner_args.ridge_fit_intercept
    protein_args.disable_esm_cache = runner_args.disable_esm_cache
    return protein_args


def _run_seed_in_process(
    protein_args: argparse.Namespace,
    shared_esm_components,
) -> None:
    from prospero.runners.run_protein import (
        logger as protein_logger,
        run_iter as run_protein_iter,
    )

    print(f"[shared-esm] starting seed {protein_args.seed}", flush=True)
    try:
        run_protein_iter(protein_args, protein_logger, shared_esm_components)
    except Exception:
        print(
            f"[shared-esm] seed {protein_args.seed} raised an exception",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
    print(f"[shared-esm] finished seed {protein_args.seed}", flush=True)


def main() -> None:
    frozen_esm_surrogate_archs = {
        "frozen_esm_mlp",
        "frozen_esm_cnn",
        "frozen_esm_flat_linear",
        "frozen_esm_flat_ridge",
    }
    parser = argparse.ArgumentParser(
        description="Run variable-n_queries seeds with a shared Python driver."
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--task", default="AAV")
    parser.add_argument(
        "--surrogate-arch",
        default="cnn",
        choices=(
            "cnn",
            "frozen_esm_mlp",
            "frozen_esm_cnn",
            "frozen_esm_flat_linear",
            "frozen_esm_flat_ridge",
        ),
    )
    parser.add_argument("--esm-cnn-projection-dim", type=int, default=None)
    parser.add_argument("--esm-cnn-use-layernorm", action="store_true", default=False)
    parser.add_argument("--esm-cnn-concat-one-hot", action="store_true", default=False)
    parser.add_argument(
        "--esm-model-name",
        default="facebook/esm2_t6_8M_UR50D",
    )
    parser.add_argument("--esm-max-length", type=int, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--ridge-fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--disable-esm-cache", action="store_true", default=False)
    parser.add_argument("--n-samples", default="8,16,32,64,128")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--min-corruptions", type=int, default=3)
    parser.add_argument("--max-corruptions", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument(
        "--esm-run-cache-base-dir",
        default="/home/lamsade/mbolanos/tmp",
    )
    parser.add_argument("--n-queries-base", type=int, default=None)
    parser.add_argument("--uv-cache-dir", default=os.environ.get("UV_CACHE_DIR"))
    args = parser.parse_args()

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_variable_k.log"
    with open(log_path, "a", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print(f"Logging console output to {log_path}")

            n_samples_values = parse_int_list(args.n_samples)
            seeds = parse_int_list(args.seeds)
            max_workers = max(1, args.max_workers)
            n_iters = args.n_iters
            run_cache_scope = (
                Path(args.esm_run_cache_base_dir)
                / f"run_variable_k_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            )
            run_cache_scope.mkdir(parents=True, exist_ok=True)
            share_frozen_esm = args.surrogate_arch in frozen_esm_surrogate_archs
            if share_frozen_esm:
                print(
                    f"[shared-esm] enabled by default for {args.surrogate_arch}",
                    flush=True,
                )
            print(
                f"[shared-esm] run cache scope: {run_cache_scope}",
                flush=True,
            )

            total_runs = len(n_samples_values) * len(seeds)
            with tqdm(
                total=total_runs,
                desc="variable_k seeds",
                unit="seed",
                leave=True,
            ) as progress:
                for n_samples in n_samples_values:
                    batch_dir = results_dir / f"n_samples_{n_samples}"
                    batch_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Running {args.task} seeds for n_samples={n_samples}")

                    cmd_template = [
                        sys.executable,
                        "src/prospero/runners/run_protein.py",
                        "--task",
                        args.task,
                        "--results_dirpath",
                        str(batch_dir),
                        "--n_iters",
                        str(n_iters),
                        "--min_corruptions",
                        str(args.min_corruptions),
                        "--max_corruptions",
                        str(args.max_corruptions),
                        "--surrogate_arch",
                        args.surrogate_arch,
                        "--full_deterministic",
                    ]
                    if args.esm_cnn_projection_dim is not None:
                        cmd_template.extend(
                            [
                                "--esm_cnn_projection_dim",
                                str(args.esm_cnn_projection_dim),
                            ]
                        )
                    if args.esm_cnn_use_layernorm:
                        cmd_template.append("--esm_cnn_use_layernorm")
                    if args.esm_cnn_concat_one_hot:
                        cmd_template.append("--esm_cnn_concat_one_hot")
                    cmd_template.extend(
                        ["--esm-run-cache-base-dir", str(run_cache_scope)]
                    )
                    n_queries = (
                        args.n_queries_base
                        if args.n_queries_base is not None
                        else n_samples
                    )
                    cmd_template.extend(["--n_queries", str(n_queries)])
                    cmd_template.extend(["--ridge_alpha", str(args.ridge_alpha)])
                    if not args.ridge_fit_intercept:
                        cmd_template.append("--no-ridge_fit_intercept")
                    if args.disable_esm_cache:
                        cmd_template.append("--disable-esm-cache")

                    env = os.environ.copy()
                    if args.uv_cache_dir:
                        env["UV_CACHE_DIR"] = args.uv_cache_dir

                    errors: list[tuple[Sequence[str], Exception]] = []
                    seed_cmds = [cmd_template + ["--seed", str(seed)] for seed in seeds]
                    if share_frozen_esm:
                        executor_workers = max_workers
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=executor_workers
                        ) as executor:
                            future_to_cmd = {}
                            shared_esm_components = None
                            if args.uv_cache_dir:
                                os.environ["UV_CACHE_DIR"] = args.uv_cache_dir
                            print(
                                f"[shared-esm] preparing shared ESM for {args.surrogate_arch}",
                                flush=True,
                            )
                            shared_esm_components = prepare_shared_esm_components(args)
                            print("[shared-esm] shared ESM ready", flush=True)
                            for seed in seeds:
                                protein_args = _build_protein_args(
                                    runner_args=args,
                                    batch_dir=batch_dir,
                                    n_iters=n_iters,
                                    n_queries=n_queries,
                                    seed=seed,
                                )
                                print(
                                    f"[shared-esm] submitting seed {seed}",
                                    flush=True,
                                )
                                future = executor.submit(
                                    _run_seed_in_process,
                                    deepcopy(protein_args),
                                    shared_esm_components,
                                )
                                future_to_cmd[future] = (
                                    f"in-process run_protein seed={seed}",
                                )
                            try:
                                for future in concurrent.futures.as_completed(future_to_cmd):
                                    cmd = future_to_cmd[future]
                                    try:
                                        future.result()
                                    except (
                                        Exception
                                    ) as exc:  # include subprocess.CalledProcessError
                                        errors.append((cmd, exc))
                                    finally:
                                        progress.update(1)
                            finally:
                                if (
                                    shared_esm_components is not None
                                    and shared_esm_components.esm_batch_worker is not None
                                ):
                                    shared_esm_components.esm_batch_worker.close()
                    else:
                        executor_workers = max_workers
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=executor_workers
                        ) as executor:
                            future_to_cmd = {}
                            future_to_cmd = {
                                executor.submit(run_seed, tuple(cmd), env): tuple(cmd)
                                for cmd in seed_cmds
                            }
                            for future in concurrent.futures.as_completed(future_to_cmd):
                                cmd = future_to_cmd[future]
                                try:
                                    future.result()
                                except (
                                    Exception
                                ) as exc:  # include subprocess.CalledProcessError
                                    errors.append((cmd, exc))
                                finally:
                                    progress.update(1)

                    if errors:
                        for cmd, exc in errors:
                            print(
                                "Seed command failed:", " ".join(cmd), file=sys.stderr
                            )
                            print(exc, file=sys.stderr)
                        raise SystemExit(1)

                    print(f"Completed seeds for n_samples={n_samples}, running ETL")
                    etl_cmd = [
                        sys.executable,
                        "src/prospero/runners/etl_results.py",
                        "--task",
                        args.task,
                        "--results_dirpath",
                        str(batch_dir),
                        "--n_iters",
                        str(n_iters),
                    ]
                    _run_command(etl_cmd, env)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
