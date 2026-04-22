from types import SimpleNamespace

import numpy as np
import torch

from prospero.search_methods import LinearGreedySearchMethod


class FakeOneHotLinearModel:
    def __init__(self, *, seq_length: int, alphabet: str, preferred_residues: list[str]):
        n_features = seq_length * len(alphabet)
        coef = np.zeros(n_features, dtype=np.float64)
        aa_to_idx = {aa: idx for idx, aa in enumerate(alphabet)}
        for position, residue in enumerate(preferred_residues):
            feature_idx = position * len(alphabet) + aa_to_idx[residue]
            coef[feature_idx] = 1.0 + position * 0.01
        self.alphabet = alphabet
        self.regressor = SimpleNamespace(coef_=coef)
        self.scaler = SimpleNamespace(scale_=np.ones_like(coef))
        self._is_fitted = True


class FakeProxy:
    def __init__(self, models, preferred_residues: list[str]):
        self.models = models
        self.preferred_residues = preferred_residues

    def get_scores(self, sequences):
        scores = []
        for sequence in sequences:
            score = 0.0
            for idx, residue in enumerate(sequence):
                if residue == self.preferred_residues[idx]:
                    score += 1.0
            scores.append(score)
        return torch.tensor(scores, dtype=torch.float32)


def _hamming_distance(lhs: str, rhs: str) -> int:
    return sum(int(a != b) for a, b in zip(lhs, rhs))


def test_linear_greedy_generates_unique_mutants_with_target_mutation_counts():
    starting_sequence = "AAAAAA"
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    preferred = ["C", "D", "E", "F", "G", "H"]
    model = FakeOneHotLinearModel(
        seq_length=len(starting_sequence),
        alphabet=alphabet,
        preferred_residues=preferred,
    )
    proxy = FakeProxy([model], preferred_residues=preferred)

    method = LinearGreedySearchMethod()
    candidates = method.generate_candidates(
        proxy=proxy,
        starting_sequence=starting_sequence,
        n_queries=8,
        ref_sequences=[starting_sequence],
    )

    assert len(candidates) == 8
    assert len(set(candidates)) == 8
    assert starting_sequence not in set(candidates)

    mutation_counts = [_hamming_distance(starting_sequence, sequence) for sequence in candidates]
    assert mutation_counts.count(1) == 4
    assert mutation_counts.count(2) == 2
    assert mutation_counts.count(3) == 2


def test_linear_greedy_respects_charge_like_substitution_map():
    starting_sequence = "RDAAAA"
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    preferred = ["W", "R", "E", "F", "G", "H"]
    model = FakeOneHotLinearModel(
        seq_length=len(starting_sequence),
        alphabet=alphabet,
        preferred_residues=preferred,
    )
    proxy = FakeProxy([model], preferred_residues=preferred)
    substitution_map = {
        "R": ["R", "K", "H"],
        "D": ["D", "E"],
        "A": list(alphabet),
    }

    method = LinearGreedySearchMethod(substitution_map=substitution_map)
    candidates = method.generate_candidates(
        proxy=proxy,
        starting_sequence=starting_sequence,
        n_queries=6,
        ref_sequences=[starting_sequence],
    )

    assert candidates
    for sequence in candidates:
        if sequence[0] != starting_sequence[0]:
            assert sequence[0] in {"K", "H"}
        if sequence[1] != starting_sequence[1]:
            assert sequence[1] == "E"
