import pickle

from prospero.experiments_config import WT_SEQUENCES
from prospero.runners import plot_zero_shot_round_fitness_histograms as histograms


def test_starting_fitness_uses_wt_and_online_queries_only(tmp_path, monkeypatch):
    wild_type = WT_SEQUENCES["AAV"]
    offline_best = "A" * len(wild_type)
    round_one = "C" * len(wild_type)
    round_two = "D" * len(wild_type)
    monkeypatch.setattr(
        histograms,
        "initial_sequence_scores",
        lambda task: [(wild_type, 0.0), (offline_best, 10.0)],
    )
    result_path = tmp_path / "seed_1.pkl"
    with result_path.open("wb") as handle:
        pickle.dump(
            {
                1: {"Iter sequences": [round_one], "Iter scores": [0.5]},
                2: {"Iter sequences": [round_two], "Iter scores": [0.7]},
            },
            handle,
        )

    _, starting_scores = histograms.load_explicit_run_rounds([result_path], "AAV")

    assert starting_scores == {1: 0.0, 2: 0.5}
