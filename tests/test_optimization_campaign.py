import numpy as np
import torch

from prospero.optimization.core import (
    AMINO_ACIDS,
    CampaignState,
    SequenceGenerator,
    load_wild_type_fitness,
)


class ReferenceDataset:
    train = np.array([list("AAA"), list("AAC")])
    valid = np.array([list("AAG")])
    train_scores = np.array([1.0, 0.2])
    valid_scores = np.array([0.4])


def test_known_wt_fitness_reads_only_explicit_baseline():
    assert load_wild_type_fitness(ReferenceDataset(), "AAA") == 1.0


def test_campaign_uses_only_wt_and_online_queries():
    campaign = CampaignState("AAA", 1.0)
    assert campaign.excluded_sequences == {"AAA"}

    campaign.observe(["AAC", "AAG"], [0.8, 1.2])

    assert campaign.incumbent_sequence == "AAG"
    assert campaign.incumbent_fitness == 1.2
    assert campaign.excluded_sequences == {"AAA", "AAC", "AAG"}


def test_campaign_keeps_wt_when_queries_are_worse():
    campaign = CampaignState("AAA", 1.0)
    campaign.observe(["AAC", "AAG"], [0.8, 0.9])
    assert campaign.incumbent_sequence == "AAA"


def test_offline_non_wt_sequences_are_not_excluded():
    campaign = CampaignState("AAA", 1.0)
    assert "AAC" not in campaign.excluded_sequences


class FixedLogitModel(SequenceGenerator):
    def __init__(self, logits):
        self.covered_length = logits.shape[1]
        self.full_ids = torch.arange(len(AMINO_ACIDS))
        self.aa_index = {
            amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)
        }
        self._initialize_search_state()
        self.logits = logits
        self.calls = []

    def logits_for_sequences(self, sequences):
        self.calls.append(sequences)
        return self.logits


def make_mms_generator(logits):
    return FixedLogitModel(logits)


def test_mms_sums_mutation_log_odds_from_fixed_incumbent_context():
    logits = torch.zeros((1, 3, len(AMINO_ACIDS)))
    logits[0, 0, AMINO_ACIDS.index("C")] = 2.0
    logits[0, 1, AMINO_ACIDS.index("D")] = 3.0
    generator = make_mms_generator(logits)

    scores = generator.marginal_mutation_scores("AAA", ["AAA", "CAA", "CDA"])

    np.testing.assert_allclose(scores, [0.0, 2.0, 5.0], atol=1e-6)


def test_mms_reuses_reference_logits_for_same_incumbent():
    logits = torch.zeros((1, 3, len(AMINO_ACIDS)))
    generator = make_mms_generator(logits)

    generator.marginal_mutation_scores("AAA", ["CAA"])
    generator.marginal_mutation_scores("AAA", ["ADA"])

    assert generator.calls == [["AAA"]]


def test_shared_model_interface_uses_incumbent_mms():
    logits = torch.zeros((1, 3, len(AMINO_ACIDS)))
    logits[0, 0, AMINO_ACIDS.index("C")] = 2.0
    generator = FixedLogitModel(logits)

    np.testing.assert_allclose(
        generator.marginal_mutation_scores("AAA", ["AAA", "CAA"]),
        [0.0, 2.0],
        atol=1e-6,
    )
