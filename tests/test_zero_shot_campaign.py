import numpy as np

from prospero.runners.run_zero_shot_prosst import CampaignState, known_wt_fitness


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
