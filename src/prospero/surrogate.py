import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import logging
import threading
import time
from dataclasses import dataclass
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .representations.dataset_store import PersistentDatasetRepresentationStore
from .representations.interplm import (
    DEFAULT_INTERPLM_REPO_ID,
    SharedInterPLMSAEPool,
    build_interplm_representation_name,
    load_interplm_sae,
    mean_pool_sae_activations,
)
from .plm.universal_hf import load_universal_hf_components
from .plm.universal_hf import infer_hidden_size
from .plm.universal_hf import tokenize_protein_sequences
from .plm.evolutionaryscale import EvolutionaryScaleBackend
from .plm.evolutionaryscale import is_evolutionaryscale_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass
class _ESMBatchRequest:
    sequences: list[str]
    representation_name: str
    max_length: int | None
    expected_sequence_length: int | None
    future: Future


class InMemoryESMEmbeddingCache:
    """Thread-safe RAM-only embedding cache with get_many/set_many API."""

    def __init__(self, storage_dtype=torch.float16):
        self.storage_dtype = storage_dtype
        self._store = {}
        self._lock = threading.Lock()

    def get_many(self, sequences):
        embeddings_by_sequence = {}
        missing_sequences = []
        with self._lock:
            for sequence in sequences:
                embedding = self._store.get(sequence)
                if embedding is None:
                    missing_sequences.append(sequence)
                    continue
                embeddings_by_sequence[sequence] = embedding
        return embeddings_by_sequence, missing_sequences

    def set_many(self, sequences, embeddings):
        with self._lock:
            for sequence, embedding in zip(sequences, embeddings):
                if sequence in self._store:
                    continue
                self._store[sequence] = (
                    embedding.detach().cpu().to(self.storage_dtype).to(torch.float32)
                )

    def clear(self) -> int:
        with self._lock:
            cleared = len(self._store)
            self._store.clear()
        return cleared


class SharedInMemoryESMCachePool:
    """Provides shared persistent RAM caches across workers/models."""

    def __init__(self, storage_dtype=torch.float16):
        self.storage_dtype = storage_dtype
        self._caches = {}
        self._lock = threading.Lock()

    def get_cache(self, model_name, max_length, representation_name, scope):
        key = (
            str(model_name),
            str(max_length),
            str(representation_name),
            str(scope),
        )
        with self._lock:
            cache = self._caches.get(key)
            if cache is None:
                cache = InMemoryESMEmbeddingCache(storage_dtype=self.storage_dtype)
                self._caches[key] = cache
            return cache

class SharedESMBatchWorker:
    def __init__(
        self,
        tokenizer,
        esm,
        device,
        sequence_backend=None,
        max_batch_sequences=512,
        max_wait_ms=4.0,
    ):
        self.tokenizer = tokenizer
        self.esm = esm
        self.device = device
        self.sequence_backend = sequence_backend
        self.max_batch_sequences = max(1, int(max_batch_sequences))
        self.max_wait_seconds = max(0.0, float(max_wait_ms) / 1000.0)
        special_ids_source = tokenizer.all_special_ids if tokenizer is not None else []
        self.special_token_ids = tuple(int(x) for x in (special_ids_source or []))
        self.tokenization_mode = "auto"
        self._pending = []
        self._condition = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="shared-esm-batch-worker",
            daemon=True,
        )
        self._thread.start()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)

    def submit(
        self,
        sequences,
        representation_name,
        max_length=None,
        expected_sequence_length=None,
    ):
        request = _ESMBatchRequest(
            sequences=list(sequences),
            representation_name=representation_name,
            max_length=max_length,
            expected_sequence_length=expected_sequence_length,
            future=Future(),
        )
        with self._condition:
            if self._closed:
                request.future.set_exception(RuntimeError("Shared ESM worker is closed."))
            else:
                self._pending.append(request)
                self._condition.notify()
        return request.future

    def compute(
        self,
        sequences,
        representation_name,
        max_length=None,
        expected_sequence_length=None,
    ):
        future = self.submit(
            sequences=sequences,
            representation_name=representation_name,
            max_length=max_length,
            expected_sequence_length=expected_sequence_length,
        )
        return future.result()

    def _compatible(self, lhs: _ESMBatchRequest, rhs: _ESMBatchRequest):
        return (
            lhs.representation_name == rhs.representation_name
            and lhs.max_length == rhs.max_length
            and lhs.expected_sequence_length == rhs.expected_sequence_length
        )

    def _pop_compatible_requests(self, first: _ESMBatchRequest):
        batch = [first]
        total_sequences = len(first.sequences)
        deadline = time.monotonic() + self.max_wait_seconds
        while total_sequences < self.max_batch_sequences:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                break
            if not self._pending:
                self._condition.wait(timeout=remaining)
                if not self._pending:
                    break

            matched_index = None
            for index, candidate in enumerate(self._pending):
                if not self._compatible(first, candidate):
                    continue
                if total_sequences + len(candidate.sequences) > self.max_batch_sequences:
                    continue
                matched_index = index
                break

            if matched_index is None:
                break

            matched = self._pending.pop(matched_index)
            batch.append(matched)
            total_sequences += len(matched.sequences)
        return batch

    def _run(self):
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed and not self._pending:
                    return
                first = self._pending.pop(0)
                batch = self._pop_compatible_requests(first)

            try:
                self._process_batch(batch)
            except Exception as exc:
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)

    def _process_batch(self, batch: list[_ESMBatchRequest]):
        if not batch:
            return

        sequences = []
        sizes = []
        for request in batch:
            sizes.append(len(request.sequences))
            sequences.extend(request.sequences)

        first = batch[0]
        all_representations = []
        for start_idx in range(0, len(sequences), self.max_batch_sequences):
            chunk_sequences = sequences[start_idx : start_idx + self.max_batch_sequences]
            if self.sequence_backend is not None:
                chunk_representations = self.sequence_backend.compute_representations(
                    chunk_sequences,
                    representation_name=first.representation_name,
                    expected_sequence_length=first.expected_sequence_length,
                ).cpu()
                all_representations.append(chunk_representations)
                continue

            encoded = tokenize_protein_sequences(
                self.tokenizer,
                chunk_sequences,
                max_length=first.max_length,
                mode=self.tokenization_mode,
            )
            self.tokenization_mode = encoded["tokenization_mode"]
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            with torch.no_grad():
                sequence_outputs = forward_hidden_state_model(
                    self.esm,
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    output_hidden_states=True,
                )
                sequence_embeddings = get_last_hidden_state(sequence_outputs)
                if first.representation_name == "mean_pool_residue_embeddings_v1":
                    chunk_representations = mean_pool_residue_embeddings(
                        sequence_embeddings, attention_mask.to(self.device)
                    ).cpu()
                elif first.representation_name == "per_residue_embeddings_v1":
                    chunk_representations = extract_residue_embeddings(
                        sequence_embeddings,
                        attention_mask.to(self.device),
                        expected_sequence_length=first.expected_sequence_length,
                        input_ids=input_ids.to(self.device),
                        special_token_ids=self.special_token_ids,
                    ).cpu()
                else:
                    raise ValueError(
                        f"Unsupported representation in shared worker: {first.representation_name}"
                    )
            all_representations.append(chunk_representations)

        representations = torch.cat(all_representations, dim=0)

        start = 0
        for request, size in zip(batch, sizes):
            end = start + size
            if not request.future.done():
                request.future.set_result(representations[start:end])
            start = end


def sequence_to_one_hot(sequence, alphabet):
    alphabet_dict = {x: idx for idx, x in enumerate(alphabet)}
    one_hot = F.one_hot(
        torch.tensor([alphabet_dict[x] for x in sequence]).long(),
        num_classes=len(alphabet),
    )
    return one_hot


def sequences_to_tensor(sequences, alphabet):
    one_hots = torch.stack(
        [sequence_to_one_hot(seq, alphabet) for seq in sequences], dim=0
    )
    one_hots = torch.permute(one_hots, [0, 2, 1]).float()
    return one_hots


def normalize_sequence(sequence):
    if isinstance(sequence, str):
        return sequence
    if isinstance(sequence, np.ndarray):
        sequence = sequence.tolist()
    return "".join(str(token) for token in sequence)


def normalize_sequences(sequences):
    return [normalize_sequence(sequence) for sequence in sequences]


def mean_pool_residue_embeddings(sequence_embeddings, attention_mask):
    residue_mask = attention_mask.bool()
    residue_mask[:, 0] = False

    last_token_indices = attention_mask.sum(dim=1).clamp(min=1) - 1
    residue_mask[
        torch.arange(residue_mask.size(0), device=residue_mask.device),
        last_token_indices,
    ] = False

    residue_mask = residue_mask.unsqueeze(-1).to(sequence_embeddings.dtype)
    pooled_embeddings = (sequence_embeddings * residue_mask).sum(dim=1)
    return pooled_embeddings / residue_mask.sum(dim=1).clamp(min=1.0)


def get_last_hidden_state(outputs):
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        return outputs.hidden_states[-1]
    if isinstance(outputs, tuple) and outputs and torch.is_tensor(outputs[0]):
        return outputs[0]
    if hasattr(outputs, "keys"):
        for key in outputs.keys():
            value = outputs[key]
            if torch.is_tensor(value) and value.dim() >= 3:
                return value
    raise ValueError("Unable to extract last hidden state from model outputs.")


def forward_hidden_state_model(model, input_ids, attention_mask, output_hidden_states=False):
    is_encoder_decoder = bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))
    if is_encoder_decoder:
        encoder = getattr(model, "encoder", None)
        if encoder is None and hasattr(model, "get_encoder"):
            encoder = model.get_encoder()
        if encoder is not None:
            return encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )


def extract_residue_embeddings(
    sequence_embeddings,
    attention_mask,
    expected_sequence_length=None,
    input_ids=None,
    special_token_ids=None,
    tokenizer=None,
):
    if special_token_ids is None and tokenizer is not None:
        special_token_ids = set(int(x) for x in (tokenizer.all_special_ids or []))
    elif special_token_ids is not None:
        special_token_ids = set(int(x) for x in special_token_ids)

    residue_embeddings = []
    for row_idx, (embedding, mask) in enumerate(zip(sequence_embeddings, attention_mask)):
        residues = None
        if input_ids is not None and special_token_ids is not None:
            valid_positions = torch.nonzero(mask.bool(), as_tuple=False).squeeze(-1)
            token_ids = input_ids[row_idx, valid_positions]
            keep_mask = torch.tensor(
                [int(tok_id.item()) not in special_token_ids for tok_id in token_ids],
                device=valid_positions.device,
                dtype=torch.bool,
            )
            kept_positions = valid_positions[keep_mask]
            if kept_positions.numel() > 0:
                residues = embedding[kept_positions]

        if residues is None:
            residue_count = int(mask.sum().item()) - 2
            residues = embedding[1 : residue_count + 1]
        if (
            expected_sequence_length is not None
            and residues.shape[0] != expected_sequence_length
        ):
            raise ValueError(
                "Per-residue ESM embeddings must match the surrogate sequence "
                "length. Check that sequences are fixed length and that "
                "esm_max_length does not truncate residues."
            )
        residue_embeddings.append(residues)

    if not residue_embeddings:
        hidden_size = sequence_embeddings.shape[-1]
        sequence_length = expected_sequence_length or 0
        return torch.empty(
            (0, sequence_length, hidden_size),
            device=sequence_embeddings.device,
            dtype=sequence_embeddings.dtype,
        )

    reference_length = expected_sequence_length or residue_embeddings[0].shape[0]
    if any(residues.shape[0] != reference_length for residues in residue_embeddings):
        raise ValueError(
                "Per-residue ESM CNN expects all sequences in a batch to have the "
                "same residue length."
        )
    return torch.stack(residue_embeddings, dim=0)


def get_esm_embedding_cache_path():
    cache_dir = Path(__file__).resolve().parents[2] / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "esm_embeddings"


def get_persistent_dataset_store_root():
    return get_esm_embedding_cache_path().parent / "dataset_representations"

class TorchModel:
    def __init__(self, args, alphabet, net, **kwargs):
        self.args = args
        self.alphabet = alphabet
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net.to(self.device)
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.loss_func = torch.nn.MSELoss()

    def get_data_loader(self, sequences, labels, shuffle):
        one_hots = sequences_to_tensor(sequences, self.alphabet).float()
        labels = torch.from_numpy(labels).float()
        dataset = torch.utils.data.TensorDataset(one_hots, labels)
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.args.proxy_batch_size, shuffle=shuffle
        )
        return loader

    def compute_loss(self, data):
        one_hots, labels = data
        outputs = torch.squeeze(self.net(one_hots.to(self.device)), dim=-1)
        loss = self.loss_func(outputs, labels.to(self.device))
        return loss

    def train(self, dataset):
        loader_train = self.get_data_loader(
            dataset.train, dataset.train_scores, shuffle=True
        )
        loader_val = self.get_data_loader(
            dataset.valid, dataset.valid_scores, shuffle=False
        )

        best_loss = np.inf
        num_no_improvement = 0

        for epoch in range(self.args.num_model_max_epochs):
            self.net.train()
            for data in loader_train:
                loss = self.compute_loss(data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if not (epoch + 1) % self.args.epochs_per_valid:
                self.net.eval()
                valid_losses = []
                with torch.no_grad():
                    for val_data in loader_val:
                        loss = self.compute_loss(val_data)
                        valid_losses.append(loss.item())
                current_loss = np.mean(valid_losses)
                if current_loss < best_loss:
                    best_loss = current_loss
                    num_no_improvement = 0
                else:
                    num_no_improvement += 1

                if num_no_improvement >= self.args.patience:
                    break

    def get_fitness(self, sequences):
        self.net.eval()
        with torch.no_grad():
            one_hots = sequences_to_tensor(sequences, self.alphabet).to(self.device)
            predictions = self.net(one_hots).squeeze()
        return predictions


class SequenceRepresentationDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_representations, labels, sequences):
        self.sequence_representations = sequence_representations
        self.labels = labels
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return (
            self.sequence_representations[index],
            self.labels[index],
            self.sequences[index],
        )


class SequenceLabelDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


class CNN(nn.Module):
    """
    The CNN architecture is adopted from the following paper with slight modification:
    - "AdaLead: A simple and robust adaptive greedy search algorithm for sequence design"
      Sam Sinai, Richard Wang, Alexander Whatley, Stewart Slocum, Elina Locane, Eric D. Kelsic
      arXiv preprint 2010.02141 (2020)
      https://arxiv.org/abs/2010.02141
    """

    def __init__(
        self,
        num_input_channels,
        seq_length,
        num_filters=32,
        hidden_dim=128,
        kernel_size=5,
    ):
        super().__init__()
        self.conv_1 = nn.Conv1d(
            num_input_channels, num_filters, kernel_size, padding="valid"
        )
        self.conv_2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding="same")
        self.conv_3 = nn.Conv1d(num_filters, num_filters, kernel_size, padding="same")
        self.global_max_pool = nn.MaxPool1d(kernel_size=seq_length - 4)
        self.dense_1 = nn.Linear(num_filters, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_1 = nn.Dropout(0.25)
        self.dense_3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Input:  [batch_size, num_input_channels, sequence_length]
        # Output: [batch_size, 1]

        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x = torch.squeeze(self.global_max_pool(x), dim=-1)
        x = torch.relu(self.dense_1(x))
        x = torch.relu(self.dense_2(x))
        x = self.dropout_1(x)
        x = self.dense_3(x)
        return x


class ESMCNNInputAdapter(nn.Module):
    def __init__(
        self,
        embedding_dim,
        output_dim=None,
        use_layernorm=False,
        concat_one_hot=False,
        one_hot_dim=20,
    ):
        super().__init__()
        self.concat_one_hot = concat_one_hot
        self.one_hot_dim = one_hot_dim
        self.layernorm = nn.LayerNorm(embedding_dim) if use_layernorm else nn.Identity()
        if output_dim is None:
            self.projection = nn.Identity()
            self.output_dim = embedding_dim
        else:
            self.projection = nn.Linear(embedding_dim, output_dim)
            self.output_dim = output_dim

    def forward(self, sequence_representations, one_hots=None):
        x = self.layernorm(sequence_representations)
        x = self.projection(x)
        if self.concat_one_hot:
            if one_hots is None:
                raise ValueError("One-hot inputs are required when concat_one_hot=True.")
            x = torch.cat([x, one_hots], dim=-1)
        return torch.permute(x, [0, 2, 1]).contiguous()


class ESMMeanPooledRegressor(nn.Module):
    def __init__(
        self,
        embedding_dim,
        mlp_hidden_dim=128,
        mlp_dropout=0.25,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, pooled_sequence_embeddings):
        return self.mlp(pooled_sequence_embeddings)


class LowRankPositionalRegressor(nn.Module):
    def __init__(self, seq_length: int, embedding_dim: int, rank: int) -> None:
        super().__init__()
        self.seq_length = int(seq_length)
        self.embedding_dim = int(embedding_dim)
        self.rank = int(rank)
        self.position_factors = nn.Parameter(torch.zeros(self.seq_length, self.rank))
        self.feature_projection = nn.Parameter(
            torch.randn(self.rank, self.embedding_dim) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("feature_mean", torch.zeros(self.embedding_dim))
        self.register_buffer("feature_std", torch.ones(self.embedding_dim))

    @torch.no_grad()
    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = mean.to(device=self.feature_mean.device, dtype=self.feature_mean.dtype)
        std = std.to(device=self.feature_std.device, dtype=self.feature_std.dtype)
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(torch.clamp(std, min=1e-6))

    def forward(self, residue_embeddings: torch.Tensor) -> torch.Tensor:
        z = (residue_embeddings - self.feature_mean[None, None, :]) / self.feature_std[
            None, None, :
        ]
        projected = torch.einsum("bld,rd->blr", z, self.feature_projection)
        scores = (projected * self.position_factors[None, :, :]).sum(dim=(1, 2))
        return (scores + self.bias).unsqueeze(-1)


class Ensemble:
    def __init__(self, models):
        self.models = models

    def train(self, dataset):
        logger.info(f"Starting training on {len(dataset.train.tolist())} samples")
        for model in self.models:
            model.train(dataset)

    @torch.no_grad()
    def get_scores(self, sequences):
        return self._call_models(sequences).mean(dim=0)

    @torch.no_grad()
    def forward_with_uncertainty(self, sequences):
        outputs = self._call_models(sequences)
        return outputs.mean(dim=0), outputs.std(dim=0, unbiased=False)

    @torch.no_grad()
    def get_ucb(self, sequences, k=0.1):
        outputs = self._call_models(sequences)
        return outputs.mean(dim=0) + k * outputs.std(dim=0, unbiased=False)

    @torch.no_grad()
    def _call_models(self, x):
        return torch.stack([model.get_fitness(x) for model in self.models])


class ConvolutionalNetworkModel(TorchModel):
    def __init__(self, seq_length, args, **kwargs):
        super().__init__(
            args,
            alphabet="ACDEFGHIKLMNPQRSTVWY",
            net=CNN(num_input_channels=20, seq_length=seq_length),
        )


class FrozenESMModel:
    class CacheMode(Enum):
        TRAIN = "train"
        EVAL = "eval"

    def __init__(
        self,
        args,
        representation_name,
        build_net=None,
        tokenizer=None,
        esm=None,
        sequence_backend=None,
        cache_allowed_sequences=None,
        enable_dataset_store=False,
        **kwargs,
    ):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = args.esm_model_name
        self.max_length = args.esm_max_length
        self.esm_forward_lock = getattr(args, "esm_forward_lock", None)
        self.esm_batch_worker = getattr(args, "esm_batch_worker", None)
        self.esm_in_memory_cache_pool = getattr(args, "esm_in_memory_cache_pool", None)
        if self.esm_in_memory_cache_pool is None:
            self.esm_in_memory_cache_pool = SharedInMemoryESMCachePool(
                storage_dtype=torch.float16
            )
        from transformers import AutoModel, AutoTokenizer  # type: ignore[reportMissingImports]

        self.sequence_backend = sequence_backend or getattr(args, "esm_sequence_backend", None)
        if self.sequence_backend is None:
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
            self.esm = (esm or AutoModel.from_pretrained(model_name)).to(self.device)
            self.esm.eval()
            for param in self.esm.parameters():
                param.requires_grad = False
            embedding_dim = infer_hidden_size(self.esm.config)
        else:
            self.tokenizer = tokenizer
            self.esm = esm
            embedding_dim = int(self.sequence_backend.hidden_size)
        self.embedding_dim = embedding_dim
        self.representation_name = representation_name
        self.model_name = model_name
        special_ids_source = (
            self.tokenizer.all_special_ids
            if self.tokenizer is not None and hasattr(self.tokenizer, "all_special_ids")
            else []
        )
        self.special_token_ids = tuple(int(x) for x in (special_ids_source or []))
        self.tokenization_mode = getattr(args, "hf_tokenization_mode", "auto")
        self.persistent_embedding_cache = self.esm_in_memory_cache_pool.get_cache(
            model_name=model_name,
            max_length=self.max_length,
            representation_name=representation_name,
            scope="persistent",
        )
        self.cache_enabled = not getattr(args, "disable_esm_cache", False)
        self.cache_allowed_sequences = (
            set(cache_allowed_sequences) if cache_allowed_sequences is not None else None
        )
        self.cache_allowed_sequences_ordered = list(
            getattr(args, "cache_allowed_sequences_ordered", [])
        )
        self.dataset_cache_task = getattr(args, "dataset_cache_task", None)
        self.dataset_fingerprint = (
            PersistentDatasetRepresentationStore.dataset_fingerprint(
                self.cache_allowed_sequences_ordered
            )
            if self.cache_allowed_sequences_ordered
            else None
        )
        self.dataset_representation_store = (
            PersistentDatasetRepresentationStore(get_persistent_dataset_store_root())
            if enable_dataset_store
            and self.dataset_cache_task is not None
            and self.dataset_fingerprint is not None
            else None
        )
        self._persistent_store_checked = False
        self.net = None
        self.optimizer = None
        self.loss_func = None
        if build_net is not None:
            self.net = build_net(embedding_dim).to(self.device)
            trainable_parameters = [
                param for param in self.net.parameters() if param.requires_grad
            ]
            self.optimizer = torch.optim.Adam(
                trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
            )
            self.loss_func = torch.nn.MSELoss()

    def get_data_loader(self, sequences, labels, shuffle):
        normalized_sequences = normalize_sequences(sequences)
        sequence_representations = self.get_sequence_representations(
            normalized_sequences,
            cache_mode=FrozenESMModel.CacheMode.EVAL,
        )
        labels = torch.from_numpy(labels).float()
        dataset = torch.utils.data.TensorDataset(sequence_representations, labels)
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.args.proxy_batch_size, shuffle=shuffle
        )
        return loader

    def _tokenize_batch(self, sequences):
        if self.sequence_backend is not None:
            raise RuntimeError(
                "_tokenize_batch should not be called when using EvolutionaryScale backend."
            )
        encoded = tokenize_protein_sequences(
            self.tokenizer,
            list(sequences),
            max_length=self.max_length,
            mode=self.tokenization_mode,
        )
        self.tokenization_mode = encoded["tokenization_mode"]
        return encoded["input_ids"], encoded["attention_mask"]

    def _compute_sequence_representations(self, sequences):
        raise NotImplementedError

    def _esm_outputs(self, input_ids, attention_mask, output_hidden_states=False):
        if self.sequence_backend is not None:
            raise RuntimeError(
                "_esm_outputs should not be called when using EvolutionaryScale backend."
            )
        model_input_ids = input_ids.to(self.device)
        model_attention_mask = attention_mask.to(self.device)
        if self.esm_forward_lock is None:
            return forward_hidden_state_model(
                self.esm,
                input_ids=model_input_ids,
                attention_mask=model_attention_mask,
                output_hidden_states=output_hidden_states,
            )

        with self.esm_forward_lock:
            return forward_hidden_state_model(
                self.esm,
                input_ids=model_input_ids,
                attention_mask=model_attention_mask,
                output_hidden_states=output_hidden_states,
            )

    def _esm_forward(self, input_ids, attention_mask):
        outputs = self._esm_outputs(
            input_ids,
            attention_mask,
            output_hidden_states=True,
        )
        return get_last_hidden_state(outputs)

    def _maybe_preload_persistent_dataset_store(self, sequences):
        if (
            self._persistent_store_checked
            or self.dataset_representation_store is None
            or not sequences
        ):
            return
        loaded, _ = self.dataset_representation_store.load_many(
            task=self.dataset_cache_task,
            model_name=self.model_name,
            max_length=self.max_length,
            representation_name=self.representation_name,
            dataset_fingerprint=self.dataset_fingerprint,
            sequences=self.cache_allowed_sequences_ordered,
        )
        if loaded:
            self.persistent_embedding_cache.set_many(
                list(loaded.keys()),
                list(loaded.values()),
            )
        self._persistent_store_checked = True

    def _maybe_write_persistent_dataset_store(self):
        if (
            self.dataset_representation_store is None
            or not self.cache_allowed_sequences_ordered
            or self.dataset_representation_store.exists(
                task=self.dataset_cache_task,
                model_name=self.model_name,
                max_length=self.max_length,
                representation_name=self.representation_name,
                dataset_fingerprint=self.dataset_fingerprint,
            )
        ):
            return
        loaded, missing = self.persistent_embedding_cache.get_many(
            self.cache_allowed_sequences_ordered
        )
        if missing:
            return
        self.dataset_representation_store.write_complete(
            task=self.dataset_cache_task,
            model_name=self.model_name,
            max_length=self.max_length,
            representation_name=self.representation_name,
            dataset_fingerprint=self.dataset_fingerprint,
            ordered_sequences=self.cache_allowed_sequences_ordered,
            embeddings=[loaded[sequence] for sequence in self.cache_allowed_sequences_ordered],
            storage_dtype="float16",
        )

    def get_sequence_representations(self, sequences, cache_mode=CacheMode.TRAIN):
        del cache_mode
        if not sequences:
            return self._compute_sequence_representations(sequences)
        if not self.cache_enabled:
            return self._compute_sequence_representations(sequences)

        start_time = time.time()
        unique_sequences = list(dict.fromkeys(sequences))
        embeddings_by_sequence = {}
        if self.cache_allowed_sequences is None:
            persistent_sequences = list(unique_sequences)
            non_persistent_sequences = []
        else:
            persistent_sequences = [
                sequence
                for sequence in unique_sequences
                if sequence in self.cache_allowed_sequences
            ]
            non_persistent_sequences = [
                sequence
                for sequence in unique_sequences
                if sequence not in self.cache_allowed_sequences
            ]

        missing_persistent_sequences = []
        if persistent_sequences:
            embeddings_by_sequence, missing_persistent_sequences = (
                self.persistent_embedding_cache.get_many(persistent_sequences)
            )
        if missing_persistent_sequences:
            self._maybe_preload_persistent_dataset_store(missing_persistent_sequences)
            loaded_from_store, missing_persistent_sequences = (
                self.persistent_embedding_cache.get_many(persistent_sequences)
            )
            embeddings_by_sequence.update(loaded_from_store)
        if missing_persistent_sequences:
            sequences_to_compute = list(missing_persistent_sequences)
            if self.cache_allowed_sequences_ordered:
                _, missing_dataset_sequences = self.persistent_embedding_cache.get_many(
                    self.cache_allowed_sequences_ordered
                )
                if missing_dataset_sequences:
                    sequences_to_compute = list(missing_dataset_sequences)
            missing_embeddings = self._compute_sequence_representations(
                sequences_to_compute
            )
            self.persistent_embedding_cache.set_many(
                sequences_to_compute, missing_embeddings
            )
            loaded_persistent_embeddings, missing_persistent_sequences = (
                self.persistent_embedding_cache.get_many(persistent_sequences)
            )
            embeddings_by_sequence.update(loaded_persistent_embeddings)
            self._maybe_write_persistent_dataset_store()

        if non_persistent_sequences:
            missing_embeddings = self._compute_sequence_representations(
                non_persistent_sequences
            )
            for sequence, embedding in zip(non_persistent_sequences, missing_embeddings):
                # Keep cache assembly device-consistent with persistent cache values (CPU).
                embeddings_by_sequence[sequence] = (
                    embedding.detach().cpu().to(torch.float32)
                )

        ordered_embeddings = [
            embeddings_by_sequence[sequence] for sequence in sequences
        ]
        stacked = torch.stack(ordered_embeddings, dim=0)
        elapsed = time.time() - start_time
        if elapsed >= 5:
            logger.info(
                "Loaded %d sequence representations in %.2fs (%d unique, %d cacheable, %d non-cacheable)",
                len(sequences),
                elapsed,
                len(unique_sequences),
                len(persistent_sequences),
                len(non_persistent_sequences),
            )
        return stacked

    def _prepare_net_inputs(self, sequence_representations, sequences=None):
        return sequence_representations

    def compute_loss(self, data):
        if len(data) == 3:
            sequence_representations, labels, sequences = data
        else:
            sequence_representations, labels = data
            sequences = None
        outputs = torch.squeeze(
            self.net(
                self._prepare_net_inputs(
                    sequence_representations, sequences=sequences
                ).to(self.device)
            ),
            dim=-1,
        )
        loss = self.loss_func(outputs, labels.to(self.device))
        return loss

    def train(self, dataset):
        loader_train = self.get_data_loader(
            dataset.train, dataset.train_scores, shuffle=True
        )
        loader_val = self.get_data_loader(
            dataset.valid, dataset.valid_scores, shuffle=False
        )

        best_loss = np.inf
        num_no_improvement = 0

        for epoch in range(self.args.num_model_max_epochs):
            if epoch == 0 or not (epoch + 1) % 10:
                logger.info("Starting epoch %d", epoch + 1)
            self.net.train()
            for batch_idx, data in enumerate(loader_train):
                if epoch == 0 and batch_idx == 0:
                    logger.info("Reached first training batch")
                loss = self.compute_loss(data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if not (epoch + 1) % self.args.epochs_per_valid:
                if epoch == 0:
                    logger.info("Reached first validation pass")
                self.net.eval()
                valid_losses = []
                with torch.no_grad():
                    for val_data in loader_val:
                        loss = self.compute_loss(val_data)
                        valid_losses.append(loss.item())
                current_loss = np.mean(valid_losses)
                if current_loss < best_loss:
                    best_loss = current_loss
                    num_no_improvement = 0
                else:
                    num_no_improvement += 1

                if num_no_improvement >= self.args.patience:
                    break

    def get_fitness(self, sequences):
        self.net.eval()
        with torch.no_grad():
            normalized_sequences = normalize_sequences(sequences)
            sequence_representations = self.get_sequence_representations(
                normalized_sequences,
                cache_mode=FrozenESMModel.CacheMode.EVAL,
            )
            predictions = self.net(
                self._prepare_net_inputs(
                    sequence_representations, sequences=normalized_sequences
                ).to(self.device),
            ).squeeze()
        return predictions


class FrozenESMMeanPooledModel(FrozenESMModel):
    def __init__(self, args, tokenizer=None, esm=None, **kwargs):
        super().__init__(
            args,
            build_net=lambda embedding_dim: ESMMeanPooledRegressor(
                embedding_dim=embedding_dim,
                mlp_hidden_dim=args.esm_mlp_hidden_dim,
                mlp_dropout=args.esm_mlp_dropout,
            ),
            representation_name="mean_pool_residue_embeddings_v1",
            tokenizer=tokenizer,
            esm=esm,
            **kwargs,
        )

    def _compute_sequence_representations(self, sequences):
        if not sequences:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        if self.sequence_backend is not None:
            return self.sequence_backend.compute_representations(
                sequences,
                representation_name="mean_pool_residue_embeddings_v1",
            )

        if self.esm_batch_worker is not None:
            return self.esm_batch_worker.compute(
                sequences=sequences,
                representation_name="mean_pool_residue_embeddings_v1",
                max_length=self.max_length,
            )

        pooled_sequence_embeddings = []
        batch_size = self.args.proxy_batch_size
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            input_ids, attention_mask = self._tokenize_batch(batch_sequences)
            with torch.no_grad():
                sequence_embeddings = self._esm_forward(input_ids, attention_mask)
                batch_embeddings = mean_pool_residue_embeddings(
                    sequence_embeddings, attention_mask.to(self.device)
                )
            pooled_sequence_embeddings.append(batch_embeddings.cpu())

        return torch.cat(pooled_sequence_embeddings, dim=0)

    def get_pooled_sequence_embeddings(self, sequences):
        return self.get_sequence_representations(sequences)


class FrozenESMPerResidueCNNModel(FrozenESMModel):
    def __init__(self, seq_length, args, tokenizer=None, esm=None, **kwargs):
        self.seq_length = seq_length
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        projection_dim = args.esm_cnn_projection_dim
        concat_one_hot = args.esm_cnn_concat_one_hot
        super().__init__(
            args,
            build_net=lambda embedding_dim: CNN(
                num_input_channels=(
                    (projection_dim or embedding_dim)
                    + (20 if concat_one_hot else 0)
                ),
                seq_length=seq_length,
            ),
            representation_name="per_residue_embeddings_v1",
            tokenizer=tokenizer,
            esm=esm,
            **kwargs,
        )
        self.input_adapter = ESMCNNInputAdapter(
            embedding_dim=self.embedding_dim,
            output_dim=projection_dim,
            use_layernorm=args.esm_cnn_use_layernorm,
            concat_one_hot=concat_one_hot,
        )

    def _compute_sequence_representations(self, sequences):
        if not sequences:
            return torch.empty(
                (0, self.seq_length, self.embedding_dim), dtype=torch.float32
            )

        if self.sequence_backend is not None:
            return self.sequence_backend.compute_representations(
                sequences,
                representation_name="per_residue_embeddings_v1",
                expected_sequence_length=self.seq_length,
            )

        if self.esm_batch_worker is not None:
            return self.esm_batch_worker.compute(
                sequences=sequences,
                representation_name="per_residue_embeddings_v1",
                max_length=self.max_length,
                expected_sequence_length=self.seq_length,
            )

        residue_sequence_embeddings = []
        batch_size = self.args.proxy_batch_size
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            input_ids, attention_mask = self._tokenize_batch(batch_sequences)
            attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                sequence_embeddings = self._esm_forward(input_ids, attention_mask)
                batch_embeddings = extract_residue_embeddings(
                    sequence_embeddings,
                    attention_mask,
                    expected_sequence_length=self.seq_length,
                    input_ids=input_ids.to(self.device),
                    tokenizer=self.tokenizer,
                )
            residue_sequence_embeddings.append(batch_embeddings.cpu())

        return torch.cat(residue_sequence_embeddings, dim=0)

    def _prepare_net_inputs(self, sequence_representations, sequences=None):
        one_hots = None
        if self.args.esm_cnn_concat_one_hot:
            if sequences is None:
                raise ValueError(
                    "Sequences are required when esm_cnn_concat_one_hot=True."
                )
            one_hots = torch.permute(
                sequences_to_tensor(sequences, self.alphabet),
                [0, 2, 1],
            ).contiguous()
        return self.input_adapter(sequence_representations, one_hots=one_hots)

    def get_data_loader(self, sequences, labels, shuffle):
        normalized_sequences = normalize_sequences(sequences)
        labels = torch.from_numpy(labels).float()
        dataset = SequenceLabelDataset(normalized_sequences, labels)
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.args.proxy_batch_size, shuffle=shuffle
        )
        return loader

    def compute_loss(self, data):
        sequences, labels = data
        batch_start = time.time()
        sequence_representations = self.get_sequence_representations(sequences)
        rep_elapsed = time.time() - batch_start
        if rep_elapsed >= 5:
            logger.info(
                "Per-residue representation load for batch of %d sequences took %.2fs",
                len(sequences),
                rep_elapsed,
            )
        return super().compute_loss((sequence_representations, labels, sequences))

    def get_residue_sequence_embeddings(self, sequences):
        return self.get_sequence_representations(sequences)


class FrozenESMFlattenedOneHotSklearnModel(FrozenESMModel):
    def __init__(
        self,
        seq_length,
        args,
        regression_type,
        include_one_hot=True,
        tokenizer=None,
        esm=None,
        **kwargs,
    ):
        self.seq_length = seq_length
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.regression_type = regression_type
        self.include_one_hot = include_one_hot
        super().__init__(
            args,
            build_net=None,
            representation_name="per_residue_embeddings_v1",
            tokenizer=tokenizer,
            esm=esm,
            enable_dataset_store=True,
            **kwargs,
        )
        self.scaler = StandardScaler()
        if regression_type == "ridge":
            self.regressor = Ridge(
                alpha=args.ridge_alpha,
                fit_intercept=args.ridge_fit_intercept,
                solver="lsqr",
            )
        elif regression_type == "linear":
            self.regressor = LinearRegression(
                fit_intercept=args.ridge_fit_intercept
            )
        else:
            raise ValueError(f"Unsupported regression_type={regression_type}")
        self._is_fitted = False

    def _compute_sequence_representations(self, sequences):
        if not sequences:
            return torch.empty(
                (0, self.seq_length, self.embedding_dim), dtype=torch.float32
            )

        if self.sequence_backend is not None:
            return self.sequence_backend.compute_representations(
                sequences,
                representation_name="per_residue_embeddings_v1",
                expected_sequence_length=self.seq_length,
            )

        if self.esm_batch_worker is not None:
            return self.esm_batch_worker.compute(
                sequences=sequences,
                representation_name="per_residue_embeddings_v1",
                max_length=self.max_length,
                expected_sequence_length=self.seq_length,
            )

        residue_sequence_embeddings = []
        batch_size = self.args.proxy_batch_size
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            input_ids, attention_mask = self._tokenize_batch(batch_sequences)
            attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                sequence_embeddings = self._esm_forward(input_ids, attention_mask)
                batch_embeddings = extract_residue_embeddings(
                    sequence_embeddings,
                    attention_mask,
                    expected_sequence_length=self.seq_length,
                    input_ids=input_ids.to(self.device),
                    tokenizer=self.tokenizer,
                )
            residue_sequence_embeddings.append(batch_embeddings.cpu())

        return torch.cat(residue_sequence_embeddings, dim=0)

    def _build_features(self, sequence_representations, sequences):
        flattened_embeddings = (
            sequence_representations.reshape(sequence_representations.shape[0], -1)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        if not self.include_one_hot:
            return flattened_embeddings

        one_hots = sequences_to_tensor(sequences, self.alphabet)
        flattened_one_hots = (
            torch.permute(one_hots, [0, 2, 1])
            .contiguous()
            .reshape(one_hots.shape[0], -1)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        return np.concatenate([flattened_embeddings, flattened_one_hots], axis=1)

    def train(self, dataset):
        train_sequences = normalize_sequences(dataset.train)
        train_representations = self.get_sequence_representations(train_sequences)
        train_features = self._build_features(train_representations, train_sequences)
        train_labels = np.asarray(dataset.train_scores, dtype=np.float32)

        scaled_train_features = self.scaler.fit_transform(train_features).astype(
            np.float32, copy=False
        )
        self.regressor.fit(scaled_train_features, train_labels)
        self._is_fitted = True

    def get_fitness(self, sequences):
        if not self._is_fitted:
            raise RuntimeError("Regressor has not been fitted. Call train() first.")

        normalized_sequences = normalize_sequences(sequences)
        sequence_representations = self.get_sequence_representations(
            normalized_sequences
        )
        features = self._build_features(sequence_representations, normalized_sequences)
        scaled_features = self.scaler.transform(features).astype(np.float32, copy=False)
        predictions = self.regressor.predict(scaled_features).astype(np.float32)
        return torch.from_numpy(predictions).to(self.device)


class InterPLMMeanPooledSklearnModel(FrozenESMModel):
    def __init__(
        self,
        seq_length,
        args,
        regression_type,
        tokenizer=None,
        esm=None,
        **kwargs,
    ):
        self.seq_length = seq_length
        self.regression_type = regression_type
        self.interplm_layer = int(args.interplm_layer)
        self.interplm_repo_id = args.interplm_repo_id
        self.interplm_normalized = args.interplm_normalized
        self.sae_token_chunk_size = int(args.sae_token_chunk_size)
        self._sae = None
        if is_evolutionaryscale_model(args.esm_model_name):
            raise ValueError(
                "interplm_mean_pool_ridge requires ESM2 hidden states and is not "
                "supported with EvolutionaryScale ESM3/ESMC models."
            )
        super().__init__(
            args,
            build_net=None,
            representation_name=build_interplm_representation_name(
                layer=self.interplm_layer,
                repo_id=self.interplm_repo_id,
                normalized=self.interplm_normalized,
            ),
            tokenizer=tokenizer,
            esm=esm,
            enable_dataset_store=True,
            **kwargs,
        )
        self.scaler = StandardScaler(copy=False)
        if regression_type == "ridge":
            self.regressor = Ridge(
                alpha=args.ridge_alpha,
                fit_intercept=args.ridge_fit_intercept,
                solver="lsqr",
            )
        elif regression_type == "linear":
            self.regressor = LinearRegression(
                fit_intercept=args.ridge_fit_intercept
            )
        else:
            raise ValueError(f"Unsupported regression_type={regression_type}")
        self._is_fitted = False

    def _get_sae(self):
        if self._sae is None:
            shared_pool = getattr(self.args, "interplm_sae_pool", None)
            if shared_pool is not None:
                self._sae = shared_pool.get(
                    plm_layer=self.interplm_layer,
                    repo_id=self.interplm_repo_id,
                    normalized=self.interplm_normalized,
                    device=self.device,
                )
            else:
                self._sae = load_interplm_sae(
                    plm_layer=self.interplm_layer,
                    repo_id=self.interplm_repo_id,
                    normalized=self.interplm_normalized,
                    map_location=self.device,
                ).to(self.device)
                self._sae.eval()
                for parameter in self._sae.parameters():
                    parameter.requires_grad = False
        return self._sae

    def _compute_sequence_representations(self, sequences):
        if not sequences:
            sae = self._get_sae()
            return torch.empty((0, sae.feature_dim), dtype=torch.float32)

        pooled_sequence_embeddings = []
        batch_size = self.args.proxy_batch_size
        sae = self._get_sae()
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            input_ids, attention_mask = self._tokenize_batch(batch_sequences)
            attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                outputs = self._esm_outputs(
                    input_ids,
                    attention_mask,
                    output_hidden_states=True,
                )
                if outputs.hidden_states is None:
                    raise RuntimeError("ESM model did not return hidden states.")
                residue_embeddings = extract_residue_embeddings(
                    outputs.hidden_states[self.interplm_layer],
                    attention_mask,
                    expected_sequence_length=self.seq_length,
                    input_ids=input_ids.to(self.device),
                    tokenizer=self.tokenizer,
                )
                batch_embeddings = mean_pool_sae_activations(
                    sae,
                    residue_embeddings,
                    token_chunk_size=self.sae_token_chunk_size,
                )
            pooled_sequence_embeddings.append(batch_embeddings.cpu())

        return torch.cat(pooled_sequence_embeddings, dim=0)

    def _build_features(self, sequence_representations):
        return (
            sequence_representations.cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def train(self, dataset):
        train_sequences = normalize_sequences(dataset.train)
        train_representations = self.get_sequence_representations(train_sequences)
        train_features = self._build_features(train_representations)
        train_labels = np.asarray(dataset.train_scores, dtype=np.float32)
        scaled_train_features = self.scaler.fit_transform(train_features)
        self.regressor.fit(scaled_train_features, train_labels)
        self._is_fitted = True

    def get_fitness(self, sequences):
        if not self._is_fitted:
            raise RuntimeError("Regressor has not been fitted. Call train() first.")
        normalized_sequences = normalize_sequences(sequences)
        sequence_representations = self.get_sequence_representations(
            normalized_sequences
        )
        features = self._build_features(sequence_representations)
        scaled_features = self.scaler.transform(features)
        predictions = self.regressor.predict(scaled_features).astype(np.float32)
        return torch.from_numpy(predictions).to(self.device)


class InterPLMLowRankPositionalModel(FrozenESMModel):
    def __init__(
        self,
        seq_length,
        args,
        tokenizer=None,
        esm=None,
        **kwargs,
    ):
        self.seq_length = int(seq_length)
        if is_evolutionaryscale_model(args.esm_model_name):
            raise ValueError(
                "interplm_low_rank_positional requires ESM2 hidden states and is "
                "not supported with EvolutionaryScale ESM3/ESMC models."
            )
        self.interplm_layer = int(args.interplm_layer)
        self.interplm_repo_id = str(args.interplm_repo_id)
        self.interplm_normalized = bool(args.interplm_normalized)
        self.sae_token_chunk_size = int(args.sae_token_chunk_size)
        self.rank = int(getattr(args, "low_rank_positional_rank", 16))
        self.low_rank_l2 = float(getattr(args, "low_rank_positional_l2", 1e-4))
        self.representation_batch_size = int(
            getattr(args, "low_rank_positional_repr_batch_size", args.proxy_batch_size)
        )
        self.low_rank_input = str(
            getattr(args, "low_rank_positional_input", "esm")
        ).strip().lower()
        if self.low_rank_input not in {"esm", "sae", "esm_sae_concat"}:
            raise ValueError(
                f"Unsupported low_rank_positional_input={self.low_rank_input!r}. "
                "Expected one of: esm, sae, esm_sae_concat."
            )
        self._sae = None
        sae_feature_dim = None
        if self.low_rank_input in {"sae", "esm_sae_concat"}:
            sae = load_interplm_sae(
                plm_layer=self.interplm_layer,
                repo_id=self.interplm_repo_id,
                normalized=self.interplm_normalized,
                map_location="cpu",
            )
            sae_feature_dim = int(sae.feature_dim)
            del sae
        self._feature_stats_ready = False
        super().__init__(
            args,
            build_net=lambda embedding_dim: LowRankPositionalRegressor(
                seq_length=self.seq_length,
                embedding_dim=(
                    int(sae_feature_dim)
                    if self.low_rank_input == "sae"
                    else (
                        int(embedding_dim) + int(sae_feature_dim)
                        if self.low_rank_input == "esm_sae_concat"
                        else int(embedding_dim)
                    )
                ),
                rank=self.rank,
            ),
            representation_name=(
                f"interplm_source_residue_embeddings_v1__layer_{self.interplm_layer}"
                if self.low_rank_input == "esm"
                else (
                    (
                        f"interplm_source_sae_residue_embeddings_v1__layer_{self.interplm_layer}"
                        f"__repo_{self.interplm_repo_id.replace('/', '__')}"
                        f"__normalized_{self.interplm_normalized}"
                    )
                    if self.low_rank_input == "sae"
                    else (
                        f"interplm_source_esm_sae_concat_residue_embeddings_v1__layer_{self.interplm_layer}"
                        f"__repo_{self.interplm_repo_id.replace('/', '__')}"
                        f"__normalized_{self.interplm_normalized}"
                    )
                )
            ),
            tokenizer=tokenizer,
            esm=esm,
            enable_dataset_store=(self.low_rank_input == "esm"),
            **kwargs,
        )
        # Keep low-rank positional representations on-device to avoid CPU<->GPU
        # transfer bottlenecks in the hot path.
        self.cache_enabled = False
        if self.low_rank_input in {"sae", "esm_sae_concat"}:
            # SAE-based inputs are high dimensional; use smaller micro-batches for
            # representation extraction to avoid OOM while keeping tensors on GPU.
            self.representation_batch_size = min(self.representation_batch_size, 16)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=float(getattr(args, "low_rank_positional_lr", args.lr)),
            weight_decay=0.0,
        )

    def _get_sae(self):
        if self._sae is None:
            shared_pool = getattr(self.args, "interplm_sae_pool", None)
            if shared_pool is not None:
                self._sae = shared_pool.get(
                    plm_layer=self.interplm_layer,
                    repo_id=self.interplm_repo_id,
                    normalized=self.interplm_normalized,
                    device=self.device,
                )
            else:
                self._sae = load_interplm_sae(
                    plm_layer=self.interplm_layer,
                    repo_id=self.interplm_repo_id,
                    normalized=self.interplm_normalized,
                    map_location=self.device,
                ).to(self.device)
                self._sae.eval()
                for parameter in self._sae.parameters():
                    parameter.requires_grad = False
        return self._sae

    def _compute_sequence_representations(self, sequences):
        if not sequences:
            return torch.empty(
                (0, self.seq_length, self.net.embedding_dim),
                dtype=torch.float32,
                device=self.device,
            )

        residue_sequence_embeddings = []
        batch_size = max(1, int(self.representation_batch_size))
        start = 0
        while start < len(sequences):
            batch_end = min(start + batch_size, len(sequences))
            batch_sequences = sequences[start:batch_end]
            try:
                input_ids, attention_mask = self._tokenize_batch(batch_sequences)
                attention_mask = attention_mask.to(self.device)
                with torch.no_grad():
                    outputs = self._esm_outputs(
                        input_ids,
                        attention_mask,
                        output_hidden_states=True,
                    )
                    if outputs.hidden_states is None:
                        raise RuntimeError("ESM model did not return hidden states.")
                    batch_embeddings = extract_residue_embeddings(
                        outputs.hidden_states[self.interplm_layer],
                        attention_mask,
                        expected_sequence_length=self.seq_length,
                        input_ids=input_ids.to(self.device),
                        tokenizer=self.tokenizer,
                    )
                    if self.low_rank_input in {"sae", "esm_sae_concat"}:
                        sae = self._get_sae()
                        batch_n, sequence_length, _ = batch_embeddings.shape
                        flattened = batch_embeddings.reshape(batch_n * sequence_length, -1)
                        projected_chunks = []
                        chunk_size = max(1, int(self.sae_token_chunk_size))
                        for chunk_start in range(0, flattened.shape[0], chunk_size):
                            chunk_stop = min(flattened.shape[0], chunk_start + chunk_size)
                            projected_chunks.append(
                                sae.encode(flattened[chunk_start:chunk_stop]).to(torch.float32)
                            )
                        sae_embeddings = torch.cat(projected_chunks, dim=0).reshape(
                            batch_n,
                            sequence_length,
                            -1,
                        )
                        if self.low_rank_input == "sae":
                            batch_embeddings = sae_embeddings
                        else:
                            batch_embeddings = torch.cat(
                                [batch_embeddings, sae_embeddings], dim=-1
                            )
                residue_sequence_embeddings.append(batch_embeddings)
                start = batch_end
            except torch.OutOfMemoryError:
                if batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    "OOM in sequence representation extraction; retrying with micro-batch size=%d",
                    batch_size,
                )

        return torch.cat(residue_sequence_embeddings, dim=0)

    def get_data_loader(self, sequences, labels, shuffle):
        normalized_sequences = normalize_sequences(sequences)
        labels_tensor = torch.from_numpy(np.asarray(labels, dtype=np.float32)).float()
        dataset = SequenceLabelDataset(normalized_sequences, labels_tensor)
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.args.proxy_batch_size, shuffle=shuffle
        )

    @torch.no_grad()
    def _prepare_feature_stats(self, train_sequences):
        if self._feature_stats_ready:
            return
        input_dim = int(getattr(self.net, "embedding_dim", self.embedding_dim))
        sums = torch.zeros(input_dim, dtype=torch.float32, device=self.device)
        sums_sq = torch.zeros(input_dim, dtype=torch.float32, device=self.device)
        total_tokens = 0
        batch_size = self.args.proxy_batch_size
        for start in range(0, len(train_sequences), batch_size):
            batch_sequences = train_sequences[start : start + batch_size]
            reps = self.get_sequence_representations(batch_sequences)
            flat = reps.reshape(-1, reps.shape[-1])
            sums += flat.sum(dim=0)
            sums_sq += (flat * flat).sum(dim=0)
            total_tokens += int(flat.shape[0])

        if total_tokens < 1:
            mean = torch.zeros(input_dim, dtype=torch.float32, device=self.device)
            std = torch.ones(input_dim, dtype=torch.float32, device=self.device)
        else:
            mean = sums / float(total_tokens)
            var = (sums_sq / float(total_tokens)) - mean * mean
            std = torch.sqrt(torch.clamp(var, min=1e-6))
        self.net.set_feature_stats(mean, std)
        self._feature_stats_ready = True

    def compute_loss(self, data):
        sequences, labels = data
        sequence_representations = self.get_sequence_representations(list(sequences))
        predictions = self.net(sequence_representations.to(self.device)).squeeze(-1)
        labels = labels.to(self.device)
        mse = F.mse_loss(predictions, labels)
        reg = self.low_rank_l2 * (
            torch.sum(self.net.position_factors * self.net.position_factors)
            + torch.sum(self.net.feature_projection * self.net.feature_projection)
        )
        return mse + reg

    def train(self, dataset):
        train_sequences = normalize_sequences(dataset.train)
        self._prepare_feature_stats(train_sequences)
        super().train(dataset)

    def get_fitness(self, sequences):
        self.net.eval()
        with torch.no_grad():
            normalized_sequences = normalize_sequences(sequences)
            if not normalized_sequences:
                return torch.empty((0,), dtype=torch.float32, device=self.device)

            predictions = []
            batch_size = self.args.proxy_batch_size
            for start in range(0, len(normalized_sequences), batch_size):
                batch_sequences = normalized_sequences[start : start + batch_size]
                batch_representations = self.get_sequence_representations(
                    batch_sequences,
                    cache_mode=FrozenESMModel.CacheMode.EVAL,
                )
                batch_predictions = self.net(batch_representations.to(self.device)).squeeze(-1)
                predictions.append(batch_predictions)
        return torch.cat(predictions, dim=0)


class OneHotSklearnModel:
    def __init__(self, seq_length, args, regression_type):
        self.seq_length = seq_length
        self.args = args
        self.regression_type = regression_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.scaler = StandardScaler()
        if regression_type == "ridge":
            self.regressor = Ridge(
                alpha=args.ridge_alpha,
                fit_intercept=args.ridge_fit_intercept,
            )
        elif regression_type == "linear":
            self.regressor = LinearRegression(
                fit_intercept=args.ridge_fit_intercept
            )
        else:
            raise ValueError(f"Unsupported regression_type={regression_type}")
        self._is_fitted = False

    def _build_features(self, sequences):
        one_hots = sequences_to_tensor(sequences, self.alphabet)
        return (
            torch.permute(one_hots, [0, 2, 1])
            .contiguous()
            .reshape(one_hots.shape[0], -1)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def train(self, dataset):
        train_sequences = normalize_sequences(dataset.train)
        train_features = self._build_features(train_sequences)
        train_labels = np.asarray(dataset.train_scores, dtype=np.float32)
        scaled_train_features = self.scaler.fit_transform(train_features)
        self.regressor.fit(scaled_train_features, train_labels)
        self._is_fitted = True

    def get_fitness(self, sequences):
        if not self._is_fitted:
            raise RuntimeError("Regressor has not been fitted. Call train() first.")
        normalized_sequences = normalize_sequences(sequences)
        features = self._build_features(normalized_sequences)
        scaled_features = self.scaler.transform(features)
        predictions = self.regressor.predict(scaled_features).astype(np.float32)
        return torch.from_numpy(predictions).to(self.device)


def build_surrogate_model(seq_length, args, shared_esm_components=None):
    tokenizer = None
    esm = None
    sequence_backend = None
    esm_forward_lock = None
    esm_batch_worker = None
    esm_in_memory_cache_pool = None
    interplm_sae_pool = None
    if shared_esm_components is not None:
        if hasattr(shared_esm_components, "tokenizer"):
            tokenizer = shared_esm_components.tokenizer
            esm = shared_esm_components.esm
            sequence_backend = getattr(shared_esm_components, "sequence_backend", None)
            esm_forward_lock = getattr(shared_esm_components, "esm_forward_lock", None)
            esm_batch_worker = getattr(shared_esm_components, "esm_batch_worker", None)
            esm_in_memory_cache_pool = getattr(
                shared_esm_components, "esm_in_memory_cache_pool", None
            )
            interplm_sae_pool = getattr(shared_esm_components, "interplm_sae_pool", None)
        elif isinstance(shared_esm_components, tuple):
            tokenizer, esm = shared_esm_components
    args.esm_forward_lock = esm_forward_lock
    args.esm_batch_worker = esm_batch_worker
    args.esm_in_memory_cache_pool = esm_in_memory_cache_pool
    args.interplm_sae_pool = interplm_sae_pool
    args.esm_sequence_backend = sequence_backend

    if args.surrogate_arch == "one_hot_ridge":
        return OneHotSklearnModel(
            seq_length,
            args,
            regression_type="ridge",
        )
    if args.surrogate_arch == "interplm_mean_pool_ridge":
        return InterPLMMeanPooledSklearnModel(
            seq_length,
            args,
            regression_type="ridge",
            tokenizer=tokenizer,
            esm=esm,
            cache_allowed_sequences=getattr(args, "cache_allowed_sequences", set()),
        )
    if args.surrogate_arch == "interplm_low_rank_positional":
        return InterPLMLowRankPositionalModel(
            seq_length,
            args,
            tokenizer=tokenizer,
            esm=esm,
            cache_allowed_sequences=getattr(args, "cache_allowed_sequences", set()),
        )

    if args.surrogate_arch in {
        "frozen_esm_mlp",
        "frozen_esm_cnn",
        "frozen_esm_flat_linear",
        "frozen_esm_flat_ridge",
        "frozen_esm_flat_ridge_no_onehot",
    }:
        if args.surrogate_arch == "frozen_esm_mlp":
            return FrozenESMMeanPooledModel(
                args,
                tokenizer=tokenizer,
                esm=esm,
                sequence_backend=sequence_backend,
                cache_allowed_sequences=getattr(args, "cache_allowed_sequences", set()),
            )
        if args.surrogate_arch == "frozen_esm_flat_linear":
            return FrozenESMFlattenedOneHotSklearnModel(
                seq_length,
                args,
                regression_type="linear",
                tokenizer=tokenizer,
                esm=esm,
                sequence_backend=sequence_backend,
                cache_allowed_sequences=getattr(args, "cache_allowed_sequences", set()),
            )
        if args.surrogate_arch == "frozen_esm_flat_ridge":
            return FrozenESMFlattenedOneHotSklearnModel(
                seq_length,
                args,
                regression_type="ridge",
                include_one_hot=True,
                tokenizer=tokenizer,
                esm=esm,
                sequence_backend=sequence_backend,
                cache_allowed_sequences=getattr(args, "cache_allowed_sequences", set()),
            )
        if args.surrogate_arch == "frozen_esm_flat_ridge_no_onehot":
            return FrozenESMFlattenedOneHotSklearnModel(
                seq_length,
                args,
                regression_type="ridge",
                include_one_hot=False,
                tokenizer=tokenizer,
                esm=esm,
                sequence_backend=sequence_backend,
                cache_allowed_sequences=getattr(args, "cache_allowed_sequences", set()),
            )
        return FrozenESMPerResidueCNNModel(
            seq_length,
            args,
            tokenizer=tokenizer,
            esm=esm,
            sequence_backend=sequence_backend,
            cache_allowed_sequences=getattr(args, "cache_allowed_sequences", set()),
        )
    return ConvolutionalNetworkModel(seq_length, args)


@dataclass(frozen=True)
class SharedESMComponents:
    tokenizer: object
    esm: object
    sequence_backend: object | None = None
    esm_forward_lock: object | None = None
    esm_batch_worker: SharedESMBatchWorker | None = None
    esm_in_memory_cache_pool: SharedInMemoryESMCachePool | None = None
    interplm_sae_pool: SharedInterPLMSAEPool | None = None


def prepare_shared_esm_components(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sequence_backend = None
    if is_evolutionaryscale_model(args.esm_model_name):
        print(f"[shared-evo] loading model={args.esm_model_name}", flush=True)
        sequence_backend = EvolutionaryScaleBackend.load(
            args.esm_model_name,
            device=device,
        )
        tokenizer = None
        esm = None
        esm_forward_lock = None
        esm_batch_worker = SharedESMBatchWorker(
            tokenizer=None,
            esm=None,
            device=device,
            sequence_backend=sequence_backend,
            max_batch_sequences=getattr(args, "shared_esm_max_batch_sequences", 64),
            max_wait_ms=getattr(args, "shared_esm_max_wait_ms", 4.0),
        )
        print(
            f"[shared-evo] loaded sdk_model={sequence_backend.sdk_name} "
            f"hidden_size={sequence_backend.hidden_size} on {device}",
            flush=True,
        )
    else:
        print(f"[shared-hf] loading model={args.esm_model_name}", flush=True)
        components = load_universal_hf_components(
            args.esm_model_name,
            device=device,
            trust_remote_code=getattr(args, "hf_trust_remote_code", True),
        )
        tokenizer = components.tokenizer
        esm = components.model
        print(
            f"[shared-hf] loaded model_type={components.model_type} "
            f"hidden_size={components.hidden_size} loader={components.model_loader_class} on {device} "
            f"(local_cache={components.local_files_only_used})",
            flush=True,
        )
        esm_forward_lock = threading.Lock()
        esm_batch_worker = SharedESMBatchWorker(
            tokenizer=tokenizer,
            esm=esm,
            device=device,
            sequence_backend=None,
            max_batch_sequences=getattr(args, "shared_esm_max_batch_sequences", 64),
            max_wait_ms=getattr(args, "shared_esm_max_wait_ms", 4.0),
        )
    esm_in_memory_cache_pool = SharedInMemoryESMCachePool(storage_dtype=torch.float16)
    interplm_sae_pool = SharedInterPLMSAEPool()
    return SharedESMComponents(
        tokenizer=tokenizer,
        esm=esm,
        sequence_backend=sequence_backend,
        esm_forward_lock=esm_forward_lock,
        esm_batch_worker=esm_batch_worker,
        esm_in_memory_cache_pool=esm_in_memory_cache_pool,
        interplm_sae_pool=interplm_sae_pool,
    )
