import numpy as np

from prospero.runners.run_plm_scoring_benchmark import marginal_mutation_scores


def test_marginal_mutation_score_uses_fixed_reference_logits():
    aa_to_col = {aa: index for index, aa in enumerate("ACDE")}
    log_probs = np.array(
        [
            [-1.0, -2.0, -3.0, -4.0],
            [-4.0, -1.0, -2.0, -3.0],
            [-3.0, -4.0, -1.0, -2.0],
        ]
    )

    scores = marginal_mutation_scores(
        "ACD", ["ACD", "CCD", "CCE"], log_probs, aa_to_col
    )

    np.testing.assert_allclose(scores, [0.0, -1.0, -2.0])
