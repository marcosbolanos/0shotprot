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
from pathlib import Path
from time import sleep

from .esm.cache import ESMEmbeddingFileCache

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


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
    def __init__(
        self,
        args,
        build_net,
        representation_name,
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
        from transformers import AutoModel, AutoTokenizer  # type: ignore[reportMissingImports]

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.esm = (esm or AutoModel.from_pretrained(model_name)).to(self.device)
        self.esm.eval()
        for param in self.esm.parameters():
            param.requires_grad = False

        embedding_dim = self.esm.config.hidden_size
        self.embedding_dim = embedding_dim
        self.embedding_cache = ESMEmbeddingFileCache(
            model_name=model_name,
            max_length=self.max_length,
            representation_name=representation_name,
            cache_root=get_esm_embedding_cache_path(),
        )
        # Full-cache mode is intentionally disabled to avoid unbounded cache growth.
        # If no whitelist is provided, no sequence is cacheable.
        self.cache_allowed_sequences = (
            set(cache_allowed_sequences) if cache_allowed_sequences is not None else set()
        )
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
            normalized_sequences
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

    def get_sequence_representations(self, sequences):
        if not sequences:
            return self._compute_sequence_representations(sequences)

        start_time = time.time()
        unique_sequences = list(dict.fromkeys(sequences))
        cacheable_sequences = [
            sequence
            for sequence in unique_sequences
            if sequence in self.cache_allowed_sequences
        ]
        non_cacheable_sequences = [
            sequence
            for sequence in unique_sequences
            if sequence not in self.cache_allowed_sequences
        ]

        embeddings_by_sequence = {}
        missing_cacheable_sequences = []
        if cacheable_sequences:
            (
                embeddings_by_sequence,
                missing_cacheable_sequences,
            ) = self.embedding_cache.get_many(cacheable_sequences)

        if missing_cacheable_sequences:
            missing_embeddings = self._compute_sequence_representations(
                missing_cacheable_sequences
            )
            self.embedding_cache.set_many(
                missing_cacheable_sequences, missing_embeddings
            )
            for sequence, embedding in zip(
                missing_cacheable_sequences, missing_embeddings
            ):
                embeddings_by_sequence[sequence] = embedding

        if non_cacheable_sequences:
            non_cacheable_embeddings = self._compute_sequence_representations(
                non_cacheable_sequences
            )
            for sequence, embedding in zip(
                non_cacheable_sequences, non_cacheable_embeddings
            ):
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
                len(cacheable_sequences),
                len(non_cacheable_sequences),
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
                normalized_sequences
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


def build_surrogate_model(seq_length, args, shared_esm_components=None):
    if args.surrogate_arch in {"frozen_esm_mlp", "frozen_esm_cnn"}:
        tokenizer = None
        esm = None
        esm_forward_lock = None
        if shared_esm_components is not None:
            tokenizer = shared_esm_components.tokenizer
            esm = shared_esm_components.esm
            esm_forward_lock = shared_esm_components.esm_forward_lock
        args.esm_forward_lock = esm_forward_lock
        if args.surrogate_arch == "frozen_esm_mlp":
            return FrozenESMMeanPooledModel(
                args,
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
    esm_forward_lock: threading.Lock


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
    return SharedESMComponents(
        tokenizer=tokenizer,
        esm=esm,
        esm_forward_lock=threading.Lock(),
    )
