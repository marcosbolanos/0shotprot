"""Run the variable-n_queries experiments from Python instead of bash."""
import argparse
import concurrent.futures
import os
import subprocess
import sys
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
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def parse_int_list(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value}") from error


def _run_command(process_args: Sequence[str], env: Optional[dict[str, str]]) -> None:
    completed = subprocess.run(
        process_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    completed.check_returncode()


def run_seed(process_args: Sequence[str], env: dict[str, str]) -> None:
    _run_command(process_args, env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run variable-n_queries seeds with a shared Python driver."
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--task", default="AAV")
    parser.add_argument(
        "--surrogate-arch", default="cnn", choices=("cnn", "frozen_esm_mlp")
    )
    parser.add_argument("--n-samples", default="8,16,32,64,128")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--min-corruptions", type=int, default=3)
    parser.add_argument("--max-corruptions", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--n-queries-base", type=int, default=None)
    parser.add_argument(
        "--uv-cache-dir", default=os.environ.get("UV_CACHE_DIR")
    )
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
                n_queries = args.n_queries_base if args.n_queries_base is not None else n_samples
                cmd_template.extend(["--n_queries", str(n_queries)])

                env = os.environ.copy()
                if args.uv_cache_dir:
                    env["UV_CACHE_DIR"] = args.uv_cache_dir

                seed_cmds = [cmd_template + ["--seed", str(seed)] for seed in seeds]

                errors: list[tuple[Sequence[str], Exception]] = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_cmd = {
                        executor.submit(run_seed, tuple(cmd), env): tuple(cmd)
                        for cmd in seed_cmds
                    }
                    for future in concurrent.futures.as_completed(future_to_cmd):
                        cmd = future_to_cmd[future]
                        try:
                            future.result()
                        except Exception as exc:  # include subprocess.CalledProcessError
                            errors.append((cmd, exc))
                        finally:
                            progress.update(1)

                if errors:
                    for cmd, exc in errors:
                        print("Seed command failed:", " ".join(cmd), file=sys.stderr)
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
