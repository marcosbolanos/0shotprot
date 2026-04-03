from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from prospero.esm.cache import ESMEmbeddingFileCache


def _namespace_size(namespace: Path) -> int:
    return sum(path.stat().st_size for path in namespace.rglob("*.pt"))


def _make_tensor(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float32)


def test_cache_hits_are_faster_than_recompute(tmp_path):
    cache = ESMEmbeddingFileCache(
        model_name="fake-esm",
        max_length=None,
        representation_name="mean_pool_residue_embeddings_v1",
        cache_root=tmp_path / "cache",
        storage_dtype="float16",
        max_disk_bytes=1_000_000_000,
        max_memory_bytes=64 * 1024 * 1024,
        eviction_check_interval=16,
    )
    sequences = [f"SEQ_{i:04d}" for i in range(80)]

    start_miss = time.perf_counter()
    _, missing = cache.get_many(sequences)
    computed = []
    for index, _sequence in enumerate(missing):
        time.sleep(0.002)
        computed.append(_make_tensor(index, (128,)))
    cache.set_many(missing, computed)
    miss_elapsed = time.perf_counter() - start_miss

    start_hit = time.perf_counter()
    loaded, missing_after = cache.get_many(sequences)
    hit_elapsed = time.perf_counter() - start_hit

    assert not missing_after
    assert len(loaded) == len(sequences)
    assert hit_elapsed < miss_elapsed * 0.35


def test_cache_is_compact_with_float16_storage(tmp_path):
    sequences = [f"SEQ_{i:03d}" for i in range(24)]
    embeddings = [_make_tensor(i, (128, 512)) for i in range(len(sequences))]

    float32_cache = ESMEmbeddingFileCache(
        model_name="fake-esm",
        max_length=None,
        representation_name="per_residue_embeddings_v1",
        cache_root=tmp_path / "float32_cache",
        storage_dtype="float32",
        max_disk_bytes=1_000_000_000,
        max_memory_bytes=0,
        eviction_check_interval=128,
    )
    float16_cache = ESMEmbeddingFileCache(
        model_name="fake-esm",
        max_length=None,
        representation_name="per_residue_embeddings_v1",
        cache_root=tmp_path / "float16_cache",
        storage_dtype="float16",
        max_disk_bytes=1_000_000_000,
        max_memory_bytes=0,
        eviction_check_interval=128,
    )

    float32_cache.set_many(sequences, embeddings)
    float16_cache.set_many(sequences, embeddings)

    size_float32 = _namespace_size(float32_cache.namespace)
    size_float16 = _namespace_size(float16_cache.namespace)

    assert size_float16 < size_float32 * 0.7


def test_cache_stays_bounded_and_handles_concurrent_reads(tmp_path):
    cache = ESMEmbeddingFileCache(
        model_name="fake-esm",
        max_length=None,
        representation_name="per_residue_embeddings_v1",
        cache_root=tmp_path / "bounded_cache",
        storage_dtype="float16",
        max_disk_bytes=320_000,
        max_memory_bytes=16 * 1024 * 1024,
        eviction_check_interval=1,
    )
    sequences = [f"SEQ_{i:04d}" for i in range(60)]
    embeddings = [_make_tensor(i, (96, 96)) for i in range(len(sequences))]
    cache.set_many(sequences, embeddings)

    disk_usage = _namespace_size(cache.namespace)
    assert disk_usage <= int(320_000 * 1.1)

    loaded, _ = cache.get_many(sequences)
    present_sequences = sorted(loaded.keys())
    assert present_sequences

    def _read_worker():
        for _ in range(100):
            batch, _missing = cache.get_many(present_sequences)
            assert len(batch) == len(present_sequences)
            for tensor in batch.values():
                assert tuple(tensor.shape) == (96, 96)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_read_worker) for _ in range(8)]
        for future in futures:
            future.result()
