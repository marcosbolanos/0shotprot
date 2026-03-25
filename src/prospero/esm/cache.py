import hashlib
import os
import tempfile
from pathlib import Path

import torch


class ESMEmbeddingFileCache:
    def __init__(self, model_name, max_length, representation_name, cache_root=None):
        if cache_root is None:
            cache_root = Path(__file__).resolve().parents[2] / ".cache" / "esm_embeddings"

        self.cache_root = Path(cache_root)
        self.model_name = model_name
        self.max_length = max_length
        self.representation_name = representation_name

        self.namespace = self.cache_root / self._safe(model_name) / self._safe(
            f"{representation_name}__maxlen_{max_length}"
        )
        self.namespace.mkdir(parents=True, exist_ok=True)

    def _safe(self, s: str) -> str:
        return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))

    def _cache_key(self, sequence: str) -> str:
        cache_input = (
            f"{self.model_name}|{self.max_length}|{self.representation_name}|{sequence}"
        )
        return hashlib.sha256(cache_input.encode("utf-8")).hexdigest()

    def _path_for_sequence(self, sequence: str) -> Path:
        key = self._cache_key(sequence)
        subdir = self.namespace / key[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{key}.pt"

    def get_many(self, sequences):
        embeddings_by_sequence = {}
        missing_sequences = []

        for sequence in sequences:
            path = self._path_for_sequence(sequence)
            if path.exists():
                try:
                    embeddings_by_sequence[sequence] = torch.load(path, map_location="cpu")
                except Exception:
                    # Corrupt cache entry: delete and recompute
                    path.unlink(missing_ok=True)
                    missing_sequences.append(sequence)
            else:
                missing_sequences.append(sequence)

        return embeddings_by_sequence, missing_sequences

    def set_many(self, sequences, embeddings):
        for sequence, embedding in zip(sequences, embeddings):
            path = self._path_for_sequence(sequence)
            if path.exists():
                continue

            path.parent.mkdir(parents=True, exist_ok=True)

            fd, tmp_path_str = tempfile.mkstemp(
                dir=path.parent,
                prefix=path.name + ".",
                suffix=".tmp",
            )
            os.close(fd)
            tmp_path = Path(tmp_path_str)

            try:
                torch.save(embedding.detach().cpu(), tmp_path)
                os.replace(tmp_path, path)   # atomic on same filesystem
            finally:
                tmp_path.unlink(missing_ok=True)
