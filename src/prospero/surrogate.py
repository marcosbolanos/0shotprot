import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import logging
import hashlib
import threading
import time
from dataclasses import dataclass
from concurrent.futures import Future
from pathlib import Path
from enum import Enum
from time import sleep
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .esm.cache import ESMEmbeddingFileCache

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


class SharedInMemoryESMCachePool:
    """Provides shared persistent/run-scoped RAM caches across workers/models."""

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
        max_batch_sequences=512,
        max_wait_ms=4.0,
    ):
        self.tokenizer = tokenizer
        self.esm = esm
        self.device = device
        self.max_batch_sequences = max(1, int(max_batch_sequences))
        self.max_wait_seconds = max(0.0, float(max_wait_ms) / 1000.0)
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
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": first.max_length is not None,
            }
            if first.max_length is not None:
                tokenizer_kwargs["max_length"] = first.max_length

            encoded = self.tokenizer(chunk_sequences, **tokenizer_kwargs)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            with torch.no_grad():
                sequence_embeddings = self.esm(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                ).last_hidden_state
                if first.representation_name == "mean_pool_residue_embeddings_v1":
                    chunk_representations = mean_pool_residue_embeddings(
                        sequence_embeddings, attention_mask.to(self.device)
                    ).cpu()
                elif first.representation_name == "per_residue_embeddings_v1":
                    chunk_representations = extract_residue_embeddings(
                        sequence_embeddings,
                        attention_mask.to(self.device),
                        expected_sequence_length=first.expected_sequence_length,
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


def extract_residue_embeddings(
    sequence_embeddings, attention_mask, expected_sequence_length=None
):
    residue_embeddings = []
    for embedding, mask in zip(sequence_embeddings, attention_mask):
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
        return outputs.mean(dim=0), outputs.std(dim=0)

    @torch.no_grad()
    def get_ucb(self, sequences, k=0.1):
        outputs = self._call_models(sequences)
        return outputs.mean(dim=0) + k * outputs.std(dim=0)

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
        cache_allowed_sequences=None,
        **kwargs,
    ):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = args.esm_model_name
        self.max_length = args.esm_max_length
        self.esm_forward_lock = getattr(args, "esm_forward_lock", None)
        self.esm_batch_worker = getattr(args, "esm_batch_worker", None)
        self.esm_in_memory_cache_pool = getattr(args, "esm_in_memory_cache_pool", None)
        from transformers import AutoModel, AutoTokenizer  # type: ignore[reportMissingImports]

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.esm = (esm or AutoModel.from_pretrained(model_name)).to(self.device)
        self.esm.eval()
        for param in self.esm.parameters():
            param.requires_grad = False

        embedding_dim = self.esm.config.hidden_size
        self.embedding_dim = embedding_dim
        if self.esm_in_memory_cache_pool is not None:
            self.persistent_embedding_cache = self.esm_in_memory_cache_pool.get_cache(
                model_name=model_name,
                max_length=self.max_length,
                representation_name=representation_name,
                scope="persistent",
            )
            self.run_embedding_cache = self.esm_in_memory_cache_pool.get_cache(
                model_name=model_name,
                max_length=self.max_length,
                representation_name=representation_name,
                scope="run_scoped",
            )
        else:
            self.persistent_embedding_cache = ESMEmbeddingFileCache(
                model_name=model_name,
                max_length=self.max_length,
                representation_name=representation_name,
                cache_root=get_esm_embedding_cache_path(),
                max_disk_bytes=0,
                max_memory_bytes=512 * 1024 * 1024,
                storage_dtype="float16",
                eviction_check_interval=256,
            )
            self.run_embedding_cache = ESMEmbeddingFileCache(
                model_name=model_name,
                max_length=self.max_length,
                representation_name=representation_name,
                cache_root=getattr(
                    args, "esm_run_cache_root", get_esm_embedding_cache_path()
                ),
                max_disk_bytes=0,
                max_memory_bytes=512 * 1024 * 1024,
                storage_dtype="float16",
                eviction_check_interval=256,
            )
        self.cache_enabled = not getattr(args, "disable_esm_cache", False)
        self.cache_allowed_sequences = (
            set(cache_allowed_sequences) if cache_allowed_sequences is not None else set()
        )
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
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": self.max_length is not None,
        }
        if self.max_length is not None:
            tokenizer_kwargs["max_length"] = self.max_length
        encoded = self.tokenizer(sequences, **tokenizer_kwargs)
        return encoded["input_ids"], encoded["attention_mask"]

    def _compute_sequence_representations(self, sequences):
        raise NotImplementedError

    def _esm_forward(self, input_ids, attention_mask):
        if self.esm_forward_lock is None:
            return self.esm(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            ).last_hidden_state

        # Keep one shared frozen ESM resident on GPU and serialize only its forward
        # pass so concurrent seed threads do not duplicate weights or thrash VRAM.
        with self.esm_forward_lock:
            return self.esm(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            ).last_hidden_state

    def get_sequence_representations(self, sequences, cache_mode=CacheMode.TRAIN):
        if not sequences:
            return self._compute_sequence_representations(sequences)
        if not self.cache_enabled:
            return self._compute_sequence_representations(sequences)
        if cache_mode == FrozenESMModel.CacheMode.EVAL:
            return self._compute_sequence_representations(sequences)

        start_time = time.time()
        unique_sequences = list(dict.fromkeys(sequences))
        embeddings_by_sequence = {}
        persistent_sequences = [
            sequence for sequence in unique_sequences if sequence in self.cache_allowed_sequences
        ]
        run_scoped_sequences = [
            sequence for sequence in unique_sequences if sequence not in self.cache_allowed_sequences
        ]

        missing_persistent_sequences = []
        if persistent_sequences:
            embeddings_by_sequence, missing_persistent_sequences = (
                self.persistent_embedding_cache.get_many(persistent_sequences)
            )
        if missing_persistent_sequences:
            missing_embeddings = self._compute_sequence_representations(
                missing_persistent_sequences
            )
            self.persistent_embedding_cache.set_many(
                missing_persistent_sequences, missing_embeddings
            )
            for sequence, embedding in zip(
                missing_persistent_sequences, missing_embeddings
            ):
                embeddings_by_sequence[sequence] = embedding

        missing_run_sequences = []
        if run_scoped_sequences:
            run_embeddings, missing_run_sequences = self.run_embedding_cache.get_many(
                run_scoped_sequences
            )
            embeddings_by_sequence.update(run_embeddings)
        if missing_run_sequences:
            missing_embeddings = self._compute_sequence_representations(
                missing_run_sequences
            )
            self.run_embedding_cache.set_many(missing_run_sequences, missing_embeddings)
            for sequence, embedding in zip(missing_run_sequences, missing_embeddings):
                embeddings_by_sequence[sequence] = embedding

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
                len(run_scoped_sequences),
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
            **kwargs,
        )
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

    def _compute_sequence_representations(self, sequences):
        if not sequences:
            return torch.empty(
                (0, self.seq_length, self.embedding_dim), dtype=torch.float32
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
        features = self._build_features(sequence_representations, normalized_sequences)
        scaled_features = self.scaler.transform(features)
        predictions = self.regressor.predict(scaled_features).astype(np.float32)
        return torch.from_numpy(predictions).to(self.device)


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
    if args.surrogate_arch == "one_hot_ridge":
        return OneHotSklearnModel(
            seq_length,
            args,
            regression_type="ridge",
        )

    if args.surrogate_arch in {
        "frozen_esm_mlp",
        "frozen_esm_cnn",
        "frozen_esm_flat_linear",
        "frozen_esm_flat_ridge",
        "frozen_esm_flat_ridge_no_onehot",
    }:
        tokenizer = None
        esm = None
        esm_forward_lock = None
        esm_batch_worker = None
        esm_in_memory_cache_pool = None
        if shared_esm_components is not None:
            if hasattr(shared_esm_components, "tokenizer"):
                tokenizer = shared_esm_components.tokenizer
                esm = shared_esm_components.esm
                esm_forward_lock = getattr(shared_esm_components, "esm_forward_lock", None)
                esm_batch_worker = getattr(shared_esm_components, "esm_batch_worker", None)
                esm_in_memory_cache_pool = getattr(
                    shared_esm_components, "esm_in_memory_cache_pool", None
                )
            elif isinstance(shared_esm_components, tuple):
                tokenizer, esm = shared_esm_components
        args.esm_forward_lock = esm_forward_lock
        args.esm_batch_worker = esm_batch_worker
        args.esm_in_memory_cache_pool = esm_in_memory_cache_pool
        if args.surrogate_arch == "frozen_esm_mlp":
            return FrozenESMMeanPooledModel(
                args,
                tokenizer=tokenizer,
                esm=esm,
                cache_allowed_sequences=getattr(args, "cache_allowed_sequences", set()),
            )
        if args.surrogate_arch == "frozen_esm_flat_linear":
            return FrozenESMFlattenedOneHotSklearnModel(
                seq_length,
                args,
                regression_type="linear",
                tokenizer=tokenizer,
                esm=esm,
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
                cache_allowed_sequences=getattr(args, "cache_allowed_sequences", set()),
            )
        return FrozenESMPerResidueCNNModel(
            seq_length,
            args,
            tokenizer=tokenizer,
            esm=esm,
            cache_allowed_sequences=getattr(args, "cache_allowed_sequences", set()),
        )
    return ConvolutionalNetworkModel(seq_length, args)


@dataclass(frozen=True)
class SharedESMComponents:
    tokenizer: object
    esm: object
    esm_forward_lock: object | None = None
    esm_batch_worker: SharedESMBatchWorker | None = None
    esm_in_memory_cache_pool: SharedInMemoryESMCachePool | None = None


def prepare_shared_esm_components(args):
    from transformers import AutoModel, AutoTokenizer  # type: ignore[reportMissingImports]

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
        max_batch_sequences=getattr(args, "shared_esm_max_batch_sequences", 512),
        max_wait_ms=getattr(args, "shared_esm_max_wait_ms", 4.0),
    )
    esm_in_memory_cache_pool = SharedInMemoryESMCachePool(storage_dtype=torch.float16)
    return SharedESMComponents(
        tokenizer=tokenizer,
        esm=esm,
        esm_forward_lock=None,
        esm_batch_worker=esm_batch_worker,
        esm_in_memory_cache_pool=esm_in_memory_cache_pool,
    )
