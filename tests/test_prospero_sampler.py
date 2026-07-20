import numpy as np

from prospero.inference import ProteinSampler


class NestedBatchTokenizer:
    all_aas = list("ACDEFGHIKLMNPQRSTVWY")

    def tokenize(self, sequences):
        assert len(sequences) == 1
        return np.asarray(
            [self.all_aas.index(amino_acid) for amino_acid in sequences[0]]
        )


def test_substitution_vocabulary_is_tokenized_as_one_sequence():
    tokenizer = NestedBatchTokenizer()
    sampler = ProteinSampler(
        model=None,
        tokenizer=tokenizer,
        substitution_alphabet={"A": ["A", "C", "D"]},
    )

    np.testing.assert_array_equal(sampler.token_clusters[0], [0, 1, 2])
