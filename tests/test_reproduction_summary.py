import pickle

import numpy as np

from prospero.runners.summarize_reproduction import summarize_paths


def test_endpoint_uncertainty_is_sample_standard_deviation(tmp_path):
    paths = []
    for seed, score in enumerate((1.0, 2.0, 3.0), start=1):
        path = tmp_path / f"seed_{seed}.pkl"
        with path.open("wb") as handle:
            pickle.dump({10: {"Best score": score}}, handle)
        paths.append(path)

    mean, standard_deviation, count = summarize_paths(paths)

    assert mean == 2.0
    assert standard_deviation == np.std([1.0, 2.0, 3.0], ddof=1)
    assert count == 3
