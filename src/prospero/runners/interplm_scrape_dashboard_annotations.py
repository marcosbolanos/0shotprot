from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from prospero.probe_benchmark.interplm_annotations import (
    parse_interplm_layer,
    select_feature_indices,
)

# Latest completed coefficient artifact found in this repo (as of 2026-04-21).
DEFAULT_COEFFICIENT_DIR = Path("outputs/interplm_full_gpu_20260414_coeffs")
DEFAULT_CACHE_PATH = DEFAULT_COEFFICIENT_DIR / "interplm_dashboard_annotation_cache.json"
DEFAULT_OUTPUT_PATH = DEFAULT_COEFFICIENT_DIR / "interplm_dashboard_annotations.csv"
DEFAULT_RETRY_LOG_PATH = DEFAULT_COEFFICIENT_DIR / "interplm_dashboard_retry.log"


@dataclass(frozen=True)
class FeatureKey:
    layer: int
    feature_id: int


class InterPLMStreamlitClient:
    def __init__(self, *, verify_ssl: bool):
        self.verify_ssl = verify_ssl
        self._ws = None
        self._page_hash: str | None = None
        self._layer_widget_id: str | None = None
        self._feature_widget_id: str | None = None
        self._view_widget_id: str | None = None
        self._active_layer: int | None = None

    def _build_ssl_context(self):
        ssl_ctx = ssl.create_default_context()
        if not self.verify_ssl:
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
        return ssl_ctx

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._page_hash = None
        self._layer_widget_id = None
        self._feature_widget_id = None
        self._view_widget_id = None
        self._active_layer = None

    async def _connect(self) -> None:
        import websockets
        from streamlit.proto.BackMsg_pb2 import BackMsg
        from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

        await self.close()
        self._ws = await websockets.connect(
            "wss://interplm.ai/_stcore/stream",
            subprotocols=["streamlit"],
            additional_headers={"Origin": "https://interplm.ai"},
            ssl=self._build_ssl_context(),
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )

        init = BackMsg()
        init.rerun_script.widget_states.widgets.extend([])
        await self._ws.send(init.SerializeToString())
        boot_msgs = await _collect_forward_messages(self._ws, ForwardMsg)

        for msg in boot_msgs:
            msg_type = msg.WhichOneof("type")
            if msg_type == "new_session":
                self._page_hash = msg.new_session.page_script_hash
            if msg_type != "delta":
                continue
            new_element = msg.delta.new_element
            element_type = new_element.WhichOneof("type")
            if element_type == "selectbox" and new_element.selectbox.label == "Select ESM embedding layer":
                self._layer_widget_id = new_element.selectbox.id
            elif element_type == "number_input" and new_element.number_input.label == "Or specify SAE feature number":
                self._feature_widget_id = new_element.number_input.id
            elif element_type == "button" and new_element.button.label == "View Protein Activations":
                self._view_widget_id = new_element.button.id

        if not all((self._ws, self._page_hash, self._layer_widget_id, self._feature_widget_id, self._view_widget_id)):
            raise RuntimeError("Could not bootstrap InterPLM Streamlit widget IDs.")

    async def _ensure_connected(self) -> None:
        ws = self._ws
        if ws is None or getattr(ws, "closed", False):
            await self._connect()

    async def _set_layer(self, layer: int) -> None:
        from streamlit.proto.BackMsg_pb2 import BackMsg
        from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
        from streamlit.proto.WidgetStates_pb2 import WidgetState

        await self._ensure_connected()
        assert self._ws is not None
        assert self._page_hash is not None
        assert self._layer_widget_id is not None

        set_layer = BackMsg()
        set_layer.rerun_script.page_script_hash = self._page_hash
        set_layer.rerun_script.widget_states.widgets.extend(
            [WidgetState(id=self._layer_widget_id, int_value=layer - 1)]
        )
        await self._ws.send(set_layer.SerializeToString())
        layer_msgs = await _collect_forward_messages(self._ws, ForwardMsg, timeout=2.5)
        for msg in layer_msgs:
            if msg.WhichOneof("type") != "delta":
                continue
            new_element = msg.delta.new_element
            if (
                new_element.WhichOneof("type") == "number_input"
                and new_element.number_input.label == "Or specify SAE feature number"
            ):
                self._feature_widget_id = new_element.number_input.id
            elif (
                new_element.WhichOneof("type") == "button"
                and new_element.button.label == "View Protein Activations"
            ):
                self._view_widget_id = new_element.button.id
        self._active_layer = layer
        if not all((self._feature_widget_id, self._view_widget_id)):
            raise RuntimeError("Could not discover layer-specific feature/button widget IDs.")

    async def fetch_annotation(self, *, layer: int, feature_id: int) -> dict[str, object]:
        from streamlit.proto.BackMsg_pb2 import BackMsg
        from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
        from streamlit.proto.WidgetStates_pb2 import WidgetState

        try:
            await self._ensure_connected()
            assert self._ws is not None
            assert self._page_hash is not None
            assert self._layer_widget_id is not None

            if self._active_layer != layer:
                await self._set_layer(layer)
            assert self._feature_widget_id is not None
            assert self._view_widget_id is not None

            clean_text = ""
            for _ in range(3):
                rerun = BackMsg()
                rerun.rerun_script.page_script_hash = self._page_hash
                rerun.rerun_script.widget_states.widgets.extend(
                    [
                        WidgetState(id=self._layer_widget_id, int_value=layer - 1),
                        WidgetState(id=self._feature_widget_id, int_value=feature_id),
                        WidgetState(id=self._view_widget_id, trigger_value=True),
                    ]
                )
                await self._ws.send(rerun.SerializeToString())
                result_msgs = await _collect_forward_messages(self._ws, ForwardMsg, timeout=2.5)

                blocks: list[str] = []
                for msg in result_msgs:
                    if msg.WhichOneof("type") != "delta":
                        continue
                    new_element = msg.delta.new_element
                    element_type = new_element.WhichOneof("type")
                    if element_type == "number_input" and new_element.number_input.label == "Or specify SAE feature number":
                        self._feature_widget_id = new_element.number_input.id
                    elif element_type == "button" and new_element.button.label == "View Protein Activations":
                        self._view_widget_id = new_element.button.id
                    if element_type == "markdown":
                        blocks.append(new_element.markdown.body)
                    elif element_type == "heading":
                        blocks.append(new_element.heading.body)
                clean_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", "\n".join(blocks))).strip()
                if f"Details on f/{feature_id}" in clean_text:
                    parsed = extract_annotation_fields(clean_text=clean_text, feature_id=feature_id)
                    parsed["raw_text"] = clean_text
                    return parsed
        except Exception:
            await self.close()
            raise

        raise RuntimeError(f"Streamlit response mismatch for requested feature {feature_id}.")


def parse_csv_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    return values or None


def parse_int_csv_list(value: str | None) -> tuple[int, ...] | None:
    parsed = parse_csv_list(value)
    if parsed is None:
        return None
    return tuple(int(v) for v in parsed)


def load_coefficient_index_rows(
    coefficient_dir: Path,
    tasks: tuple[str, ...] | None,
    configs: tuple[str, ...] | None,
    budgets: tuple[int, ...] | None,
    seeds: tuple[int, ...] | None,
) -> list[dict[str, str]]:
    index_path = coefficient_dir / "coefficient_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing coefficient index: {index_path}")

    rows: list[dict[str, str]] = []
    with open(index_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if tasks is not None and row["task"] not in tasks:
                continue
            if configs is not None and row["config"] not in configs:
                continue
            if budgets is not None and int(row["budget"]) not in budgets:
                continue
            if seeds is not None and int(row["seed"]) not in seeds:
                continue
            rows.append(row)
    return rows


def collect_feature_keys(
    coefficient_dir: Path,
    rows: Iterable[dict[str, str]],
    top_k: int,
    coefficient_threshold: float,
    include_all_nonzero: bool,
) -> list[FeatureKey]:
    features: set[FeatureKey] = set()

    for row in rows:
        layer = parse_interplm_layer(row["config"])
        if layer is None:
            continue

        coefficient_path = coefficient_dir / row["coefficient_path"]
        if not coefficient_path.exists():
            continue
        with np.load(coefficient_path) as artifact:
            coefficients = np.asarray(artifact["coefficients"], dtype=np.float64).reshape(-1)

        selected = select_feature_indices(
            coefficients=coefficients,
            top_k=top_k,
            coefficient_threshold=coefficient_threshold,
            include_all_nonzero=include_all_nonzero,
        )
        for feature_id in selected:
            features.add(FeatureKey(layer=layer, feature_id=int(feature_id)))

    return sorted(features, key=lambda item: (item.layer, item.feature_id))


def load_cache(cache_path: Path) -> dict[str, dict[str, object]]:
    if not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return payload


def save_cache(cache_path: Path, cache: dict[str, dict[str, object]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


def extract_annotation_fields(clean_text: str, feature_id: int) -> dict[str, object]:
    lm_score: float | None = None
    lm_description: str | None = None

    pattern = (
        rf"Language Model Description for f/{feature_id} \(score=([0-9.]+)\)"
        rf"\s*(.+?)\s*(?:Feature Activation Distribution for f/{feature_id}|Concepts Identified in f/{feature_id})"
    )
    match = re.search(pattern, clean_text)
    if match:
        lm_score = float(match.group(1))
        lm_description = match.group(2).strip().lstrip("*").strip()

    no_swiss = "No Swiss-Prot concepts found for this feature." in clean_text
    return {
        "lm_score": lm_score,
        "lm_description": lm_description,
        "no_swissprot_concepts": no_swiss,
    }


async def _collect_forward_messages(ws, forward_cls, timeout: float = 2.0, max_frames: int = 800):
    messages = []
    for _ in range(max_frames):
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            break
        if isinstance(raw, str):
            continue
        msg = forward_cls()
        try:
            msg.ParseFromString(raw)
        except Exception:
            continue
        messages.append(msg)
    return messages


async def fetch_single_annotation(
    *,
    layer: int,
    feature_id: int,
    verify_ssl: bool,
):
    import websockets
    from streamlit.proto.BackMsg_pb2 import BackMsg
    from streamlit.proto.WidgetStates_pb2 import WidgetState
    from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

    ssl_ctx = ssl.create_default_context()
    if not verify_ssl:
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    async with websockets.connect(
        "wss://interplm.ai/_stcore/stream",
        subprotocols=["streamlit"],
        additional_headers={"Origin": "https://interplm.ai"},
        ssl=ssl_ctx,
        max_size=None,
    ) as ws:
        init = BackMsg()
        init.rerun_script.widget_states.widgets.extend([])
        await ws.send(init.SerializeToString())
        boot_msgs = await _collect_forward_messages(ws, ForwardMsg)

        page_hash = ""
        layer_widget_id = None
        feature_widget_id = None
        view_widget_id = None

        for msg in boot_msgs:
            msg_type = msg.WhichOneof("type")
            if msg_type == "new_session":
                page_hash = msg.new_session.page_script_hash
            if msg_type != "delta":
                continue
            new_element = msg.delta.new_element
            element_type = new_element.WhichOneof("type")
            if element_type == "selectbox" and new_element.selectbox.label == "Select ESM embedding layer":
                layer_widget_id = new_element.selectbox.id
            elif element_type == "number_input" and new_element.number_input.label == "Or specify SAE feature number":
                feature_widget_id = new_element.number_input.id
            elif element_type == "button" and new_element.button.label == "View Protein Activations":
                view_widget_id = new_element.button.id

        if not all((page_hash, layer_widget_id, feature_widget_id, view_widget_id)):
            raise RuntimeError("Could not discover InterPLM Streamlit widget IDs.")

        # First rerun picks the requested layer so Streamlit materializes the
        # layer-specific feature input widget (its internal id changes by layer).
        set_layer = BackMsg()
        set_layer.rerun_script.page_script_hash = page_hash
        set_layer.rerun_script.widget_states.widgets.extend(
            [WidgetState(id=layer_widget_id, int_value=layer - 1)]
        )
        await ws.send(set_layer.SerializeToString())
        layer_msgs = await _collect_forward_messages(ws, ForwardMsg, timeout=2.5)

        for msg in layer_msgs:
            if msg.WhichOneof("type") != "delta":
                continue
            new_element = msg.delta.new_element
            if (
                new_element.WhichOneof("type") == "number_input"
                and new_element.number_input.label == "Or specify SAE feature number"
            ):
                feature_widget_id = new_element.number_input.id
            elif (
                new_element.WhichOneof("type") == "button"
                and new_element.button.label == "View Protein Activations"
            ):
                view_widget_id = new_element.button.id

        if not all((feature_widget_id, view_widget_id)):
            raise RuntimeError("Could not discover layer-specific feature/button widget IDs.")

        rerun = BackMsg()
        rerun.rerun_script.page_script_hash = page_hash
        rerun.rerun_script.widget_states.widgets.extend(
            [
                WidgetState(id=layer_widget_id, int_value=layer - 1),
                WidgetState(id=feature_widget_id, int_value=feature_id),
                WidgetState(id=view_widget_id, trigger_value=True),
            ]
        )
        await ws.send(rerun.SerializeToString())
        result_msgs = await _collect_forward_messages(ws, ForwardMsg, timeout=2.5)

    blocks: list[str] = []
    for msg in result_msgs:
        if msg.WhichOneof("type") != "delta":
            continue
        new_element = msg.delta.new_element
        element_type = new_element.WhichOneof("type")
        if element_type == "markdown":
            blocks.append(new_element.markdown.body)
        elif element_type == "heading":
            blocks.append(new_element.heading.body)

    clean_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", "\n".join(blocks))).strip()
    parsed = extract_annotation_fields(clean_text=clean_text, feature_id=feature_id)
    parsed["raw_text"] = clean_text
    return parsed


def write_output_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer",
        "feature_id",
        "lm_score",
        "lm_description",
        "no_swissprot_concepts",
        "raw_text",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape InterPLM dashboard annotations for selected SAE features.")
    parser.add_argument("--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--coefficient-threshold", type=float, default=0.0)
    parser.add_argument("--include-all-nonzero", action="store_true", default=False)
    parser.add_argument("--max-features", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--requests-per-second", type=float, default=10.0)
    parser.add_argument("--verify-ssl", action="store_true", default=False)
    parser.add_argument("--flush-cache-every", type=int, default=1)
    parser.add_argument("--attempt-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--retry-backoff-base-seconds", type=float, default=2.0)
    parser.add_argument("--retry-backoff-max-seconds", type=float, default=300.0)
    parser.add_argument("--retry-log-path", type=Path, default=DEFAULT_RETRY_LOG_PATH)
    parser.add_argument(
        "--all-features-layer",
        type=int,
        default=None,
        help="If set, scrape all feature IDs [0, layer-feature-count) for this one layer.",
    )
    parser.add_argument(
        "--layer-feature-count",
        type=int,
        default=10240,
        help="Number of feature IDs per layer when --all-features-layer is set.",
    )
    parser.add_argument(
        "--connection-mode",
        choices=("persistent", "fresh"),
        default="persistent",
        help="Use one persistent websocket session or reconnect for each feature.",
    )
    parser.add_argument(
        "--persistent-max-failures",
        type=int,
        default=5,
        help="If persistent mode fails this many times, automatically fall back to fresh mode for stability.",
    )
    return parser


async def run_scrape(args: argparse.Namespace, feature_keys: list[FeatureKey], cache: dict[str, dict[str, object]]) -> None:
    min_interval = 1.0 / args.requests_per_second
    updated = 0
    last_request_at = 0.0
    client = InterPLMStreamlitClient(verify_ssl=bool(args.verify_ssl)) if args.connection_mode == "persistent" else None
    persistent_failures = 0
    persistent_disabled = False

    try:
        for key in tqdm(feature_keys, desc=f"Fetching InterPLM annotations ({args.connection_mode})"):
            cache_key = f"{key.layer}:{key.feature_id}"
            if cache_key in cache:
                continue

            attempt = 0
            while True:
                attempt += 1
                now = time.monotonic()
                sleep_for = min_interval - (now - last_request_at)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

                try:
                    use_persistent = args.connection_mode == "persistent" and not persistent_disabled
                    if use_persistent:
                        assert client is not None
                        result = await asyncio.wait_for(
                            client.fetch_annotation(layer=key.layer, feature_id=key.feature_id),
                            timeout=float(args.attempt_timeout_seconds),
                        )
                        persistent_failures = 0
                    else:
                        result = await asyncio.wait_for(
                            fetch_single_annotation(
                                layer=key.layer,
                                feature_id=key.feature_id,
                                verify_ssl=bool(args.verify_ssl),
                            ),
                            timeout=float(args.attempt_timeout_seconds),
                        )
                    last_request_at = time.monotonic()
                    break
                except Exception as exc:
                    last_request_at = time.monotonic()
                    if args.connection_mode == "persistent" and not persistent_disabled:
                        persistent_failures += 1
                        if persistent_failures >= int(args.persistent_max_failures):
                            persistent_disabled = True
                    backoff = min(
                        float(args.retry_backoff_max_seconds),
                        float(args.retry_backoff_base_seconds) * (2 ** min(attempt - 1, 8)),
                    )
                    backoff *= 0.85 + 0.3 * random.random()
                    with open(args.retry_log_path, "a", encoding="utf-8") as handle:
                        handle.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"layer={key.layer} feature={key.feature_id} "
                            f"attempt={attempt} mode={args.connection_mode} "
                            f"persistent_disabled={persistent_disabled} "
                            f"error={type(exc).__name__}: {exc}\n"
                        )
                    if client is not None:
                        await client.close()
                    await asyncio.sleep(backoff)

            cache[cache_key] = {
                "layer": key.layer,
                "feature_id": key.feature_id,
                **result,
            }
            updated += 1
            if updated % max(1, int(args.flush_cache_every)) == 0:
                save_cache(args.cache_path, cache)
                ordered_rows = [cache[k] for k in sorted(cache.keys(), key=lambda text: tuple(map(int, text.split(":"))))]
                write_output_rows(args.output_path, ordered_rows)
    finally:
        if client is not None:
            await client.close()


def main() -> None:
    args = build_parser().parse_args()
    if args.requests_per_second <= 0:
        raise ValueError("--requests-per-second must be > 0")
    if args.attempt_timeout_seconds <= 0:
        raise ValueError("--attempt-timeout-seconds must be > 0")
    if args.retry_backoff_base_seconds <= 0:
        raise ValueError("--retry-backoff-base-seconds must be > 0")
    if args.retry_backoff_max_seconds < args.retry_backoff_base_seconds:
        raise ValueError("--retry-backoff-max-seconds must be >= --retry-backoff-base-seconds")
    if args.all_features_layer is not None:
        if args.all_features_layer < 1:
            raise ValueError("--all-features-layer must be >= 1")
        if args.layer_feature_count < 1:
            raise ValueError("--layer-feature-count must be >= 1")

    if args.all_features_layer is not None:
        feature_keys = [
            FeatureKey(layer=int(args.all_features_layer), feature_id=int(feature_id))
            for feature_id in range(int(args.layer_feature_count))
        ]
    else:
        rows = load_coefficient_index_rows(
            coefficient_dir=args.coefficient_dir,
            tasks=parse_csv_list(args.tasks),
            configs=parse_csv_list(args.configs),
            budgets=parse_int_csv_list(args.budgets),
            seeds=parse_int_csv_list(args.seeds),
        )
        feature_keys = collect_feature_keys(
            coefficient_dir=args.coefficient_dir,
            rows=rows,
            top_k=args.top_k,
            coefficient_threshold=args.coefficient_threshold,
            include_all_nonzero=args.include_all_nonzero,
        )
    if args.max_features is not None:
        feature_keys = feature_keys[: args.max_features]

    cache = load_cache(args.cache_path)
    args.retry_log_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_scrape(args, feature_keys, cache))

    save_cache(args.cache_path, cache)

    ordered_rows = [cache[k] for k in sorted(cache.keys(), key=lambda text: tuple(map(int, text.split(":"))))]
    write_output_rows(args.output_path, ordered_rows)

    print(
        json.dumps(
            {
                "coefficient_dir": str(args.coefficient_dir),
                "n_requested_features": len(feature_keys),
                "n_cached_total": len(cache),
                "cache_path": str(args.cache_path),
                "output_path": str(args.output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
