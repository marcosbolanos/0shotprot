"""Representation proxy benchmarks backed by cached frozen ESM embeddings."""

from .pipeline import DEFAULT_PROBE_SPECS, BenchmarkRunner

__all__ = ["BenchmarkRunner", "DEFAULT_PROBE_SPECS"]
