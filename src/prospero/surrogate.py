import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import logging
import hashlib
from pathlib import Path
from time import sleep
from sqlalchemy import LargeBinary, String, Integer, create_engine, select, text  # type: ignore[reportMissingImports]
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column  # type: ignore[reportMissingImports]
from sqlalchemy.exc import OperationalError  # type: ignore[reportMissingImports]
from transformers import AutoModel, AutoTokenizer  # type: ignore[reportMissingImports]

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


def get_esm_embedding_cache_path():
    cache_dir = Path(__file__).resolve().parents[2] / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "esm_embeddings.sqlite3"


class Base(DeclarativeBase):
    pass


class ESMEmbeddingCacheRow(Base):
    __tablename__ = "esm_embedding_cache"

    cache_key: Mapped[str] = mapped_column(String, primary_key=True)
    sequence: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    max_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pooling_name: Mapped[str] = mapped_column(String, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class ESMEmbeddingCache:
    def __init__(self, model_name, max_length, embedding_dim):
        self.cache_path = get_esm_embedding_cache_path()
        self.model_name = model_name
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.pooling_name = "mean_pool_residue_embeddings"
        self.engine = create_engine(
            f"sqlite:///{self.cache_path}",
            connect_args={"timeout": 30, "check_same_thread": False},
        )
        with self.engine.begin() as connection:
            connection.execute(text("PRAGMA journal_mode=WAL"))
        try:
            Base.metadata.create_all(self.engine)
        except OperationalError as exc:
            if "already exists" not in str(exc):
                raise

    def _cache_key(self, sequence):
        cache_input = (
            f"{self.model_name}|{self.max_length}|{self.pooling_name}|{sequence}"
        )
        return hashlib.sha256(cache_input.encode("utf-8")).hexdigest()

    def get_many(self, sequences):
        if not sequences:
            return {}, []

        cache_keys = [self._cache_key(sequence) for sequence in sequences]
        rows_by_key = {}
        query_batch_size = 900
        with Session(self.engine) as session:
            for start in range(0, len(cache_keys), query_batch_size):
                batch_cache_keys = cache_keys[start : start + query_batch_size]
                statement = select(ESMEmbeddingCacheRow).where(
                    ESMEmbeddingCacheRow.cache_key.in_(batch_cache_keys)
                )
                batch_rows = session.scalars(statement).all()
                for row in batch_rows:
                    rows_by_key[row.cache_key] = row

        embeddings_by_sequence = {}
        missing_sequences = []
        for sequence, cache_key in zip(sequences, cache_keys):
            row = rows_by_key.get(cache_key)
            if row is None:
                missing_sequences.append(sequence)
                continue
            embedding_array = np.frombuffer(
                row.embedding_blob, dtype=np.float32
            ).astype(np.float32)
            embedding = torch.from_numpy(embedding_array.copy())
            embeddings_by_sequence[sequence] = embedding

        return embeddings_by_sequence, missing_sequences

    def set_many(self, sequences, embeddings):
        rows = []
        for sequence, embedding in zip(sequences, embeddings):
            embedding_array = embedding.detach().cpu().numpy().astype(np.float32)
            rows.append(
                ESMEmbeddingCacheRow(
                    cache_key=self._cache_key(sequence),
                    sequence=sequence,
                    model_name=self.model_name,
                    max_length=self.max_length,
                    pooling_name=self.pooling_name,
                    embedding_dim=self.embedding_dim,
                    embedding_blob=embedding_array.tobytes(),
                )
            )

        attempts = 0
        max_attempts = 5
        while True:
            try:
                with Session(self.engine) as session:
                    for row in rows:
                        session.merge(row)
                    session.commit()
                break
            except OperationalError as exc:
                attempts += 1
                if attempts >= max_attempts:
                    raise
                sleep(0.25 * attempts)


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


class FrozenESMMeanPooledModel:
    def __init__(self, args, tokenizer=None, esm=None, **kwargs):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = args.esm_model_name
        self.max_length = args.esm_max_length
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.esm = (esm or AutoModel.from_pretrained(model_name)).to(self.device)
        self.esm.eval()
        for param in self.esm.parameters():
            param.requires_grad = False

        embedding_dim = self.esm.config.hidden_size
        self.embedding_dim = embedding_dim
        self.embedding_cache = ESMEmbeddingCache(
            model_name=model_name,
            max_length=self.max_length,
            embedding_dim=embedding_dim,
        )
        self.net = ESMMeanPooledRegressor(
            embedding_dim=embedding_dim,
            mlp_hidden_dim=args.esm_mlp_hidden_dim,
            mlp_dropout=args.esm_mlp_dropout,
        ).to(self.device)
        trainable_parameters = [
            param for param in self.net.parameters() if param.requires_grad
        ]
        self.optimizer = torch.optim.Adam(
            trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
        )
        self.loss_func = torch.nn.MSELoss()

    def get_data_loader(self, sequences, labels, shuffle):
        normalized_sequences = normalize_sequences(sequences)
        pooled_sequence_embeddings = self.get_pooled_sequence_embeddings(
            normalized_sequences
        )
        labels = torch.from_numpy(labels).float()
        dataset = torch.utils.data.TensorDataset(pooled_sequence_embeddings, labels)
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

    def _compute_pooled_sequence_embeddings(self, sequences):
        if not sequences:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        pooled_sequence_embeddings = []
        batch_size = self.args.proxy_batch_size
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            input_ids, attention_mask = self._tokenize_batch(batch_sequences)
            with torch.no_grad():
                sequence_embeddings = self.esm(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                ).last_hidden_state
                batch_embeddings = mean_pool_residue_embeddings(
                    sequence_embeddings, attention_mask.to(self.device)
                )
            pooled_sequence_embeddings.append(batch_embeddings.cpu())

        return torch.cat(pooled_sequence_embeddings, dim=0)

    def get_pooled_sequence_embeddings(self, sequences):
        unique_sequences = list(dict.fromkeys(sequences))
        embeddings_by_sequence, missing_sequences = self.embedding_cache.get_many(
            unique_sequences
        )

        if missing_sequences:
            missing_embeddings = self._compute_pooled_sequence_embeddings(
                missing_sequences
            )
            self.embedding_cache.set_many(missing_sequences, missing_embeddings)
            for sequence, embedding in zip(missing_sequences, missing_embeddings):
                embeddings_by_sequence[sequence] = embedding

        ordered_embeddings = [
            embeddings_by_sequence[sequence] for sequence in sequences
        ]
        return torch.stack(ordered_embeddings, dim=0)

    def compute_loss(self, data):
        pooled_sequence_embeddings, labels = data
        outputs = torch.squeeze(
            self.net(pooled_sequence_embeddings.to(self.device)), dim=-1
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
            normalized_sequences = normalize_sequences(sequences)
            pooled_sequence_embeddings = self.get_pooled_sequence_embeddings(
                normalized_sequences
            )
            predictions = self.net(
                pooled_sequence_embeddings.to(self.device),
            ).squeeze()
        return predictions


def build_surrogate_model(seq_length, args, shared_esm_components=None):
    if args.surrogate_arch in {"frozen_esm_mlp"}:
        tokenizer = None
        esm = None
        if shared_esm_components is not None:
            tokenizer, esm = shared_esm_components
        return FrozenESMMeanPooledModel(args, tokenizer=tokenizer, esm=esm)
    return ConvolutionalNetworkModel(seq_length, args)


def prepare_shared_esm_components(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model_name)
    esm = AutoModel.from_pretrained(args.esm_model_name).to(device)
    esm.eval()
    for param in esm.parameters():
        param.requires_grad = False
    return tokenizer, esm
