import numpy as np
import torch

from prospero.runners.run_zero_shot_prosst import AA20, CampaignState, ProSSTGenerator, known_wt_fitness
from prospero.runners.run_zero_shot_evodiff import EvoDiffGenerator


class ReferenceDataset:
    train = np.array([list("AAA"), list("AAC")])
    valid = np.array([list("AAG")])
    train_scores = np.array([1.0, 0.2])
    valid_scores = np.array([0.4])


def test_known_wt_fitness_reads_only_explicit_baseline():
    assert known_wt_fitness(ReferenceDataset(), "AAA") == 1.0


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


def make_mms_generator(logits):
    generator = ProSSTGenerator.__new__(ProSSTGenerator)
    generator.covered_length = logits.shape[1]
    generator.full_ids = torch.arange(len(AA20))
    generator.aa_index = {aa: idx for idx, aa in enumerate(AA20)}
    generator._mms_cache = None
    generator.logits_for_sequences = lambda sequences: logits
    return generator


def test_mms_sums_mutation_log_odds_from_fixed_incumbent_context():
    logits = torch.zeros((1, 3, len(AA20)))
    logits[0, 0, AA20.index("C")] = 2.0
    logits[0, 1, AA20.index("D")] = 3.0
    generator = make_mms_generator(logits)

    scores = generator.marginal_mutation_scores("AAA", ["AAA", "CAA", "CDA"])

    np.testing.assert_allclose(scores, [0.0, 2.0, 5.0], atol=1e-6)


def test_mms_reuses_reference_logits_for_same_incumbent():
    calls = []
    logits = torch.zeros((1, 3, len(AA20)))
    generator = make_mms_generator(logits)
    generator.logits_for_sequences = lambda sequences: calls.append(sequences) or logits

    generator.marginal_mutation_scores("AAA", ["CAA"])
    generator.marginal_mutation_scores("AAA", ["ADA"])

    assert calls == [["AAA"]]


def test_evodiff_adapter_uses_fixed_incumbent_mms():
    logits = torch.zeros((1, 3, len(AA20)))
    logits[0, 0, AA20.index("C")] = 2.0
    generator = EvoDiffGenerator.__new__(EvoDiffGenerator)
    generator.covered_length = 3
    generator.full_ids = torch.arange(len(AA20))
    generator.aa_index = {aa: idx for idx, aa in enumerate(AA20)}
    generator._mms_cache = None
    generator.logits_for_sequences = lambda sequences: logits

    np.testing.assert_allclose(
        generator.marginal_mutation_scores("AAA", ["AAA", "CAA"]),
        [0.0, 2.0],
        atol=1e-6,
    )
