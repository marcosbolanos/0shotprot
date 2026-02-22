import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import logging
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


class ESMTransformerRegressor(nn.Module):
    def __init__(
        self,
        model_name,
        num_attention_heads=4,
        attention_dropout=0.1,
        mlp_hidden_dim=256,
        mlp_dropout=0.25,
    ):
        super().__init__()
        self.esm = AutoModel.from_pretrained(model_name)
        for param in self.esm.parameters():
            param.requires_grad = False

        embedding_dim = self.esm.config.hidden_size
        self.sequence_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.post_attention_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            token_embeddings = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        key_padding_mask = attention_mask == 0
        attended_embeddings, _ = self.sequence_attention(
            token_embeddings,
            token_embeddings,
            token_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended_embeddings = self.post_attention_norm(
            attended_embeddings + token_embeddings
        )
        bos_embedding = attended_embeddings[:, 0, :]
        return self.mlp(bos_embedding)


class SequenceRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


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


class FrozenESMTransformerModel:
    def __init__(self, args, **kwargs):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = args.esm_model_name
        self.max_length = args.esm_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.net = ESMTransformerRegressor(
            model_name=model_name,
            num_attention_heads=args.esm_attention_heads,
            attention_dropout=args.esm_attention_dropout,
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
        labels = torch.from_numpy(labels).float()
        dataset = SequenceRegressionDataset(normalized_sequences, labels)
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

    def compute_loss(self, data):
        sequences, labels = data
        input_ids, attention_mask = self._tokenize_batch(sequences)
        outputs = torch.squeeze(
            self.net(
                input_ids.to(self.device),
                attention_mask.to(self.device),
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
            input_ids, attention_mask = self._tokenize_batch(normalized_sequences)
            predictions = self.net(
                input_ids.to(self.device),
                attention_mask.to(self.device),
            ).squeeze()
        return predictions


def build_surrogate_model(seq_length, args):
    if args.surrogate_arch == "esm_transformer":
        return FrozenESMTransformerModel(args)
    return ConvolutionalNetworkModel(seq_length, args)
