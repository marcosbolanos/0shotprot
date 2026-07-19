import numpy as np

from prospero.runners.run_plm_full_pll_alignment import mutation_log_odds


def test_mutation_log_odds_uses_fixed_reference_logits():
    aa_to_col = {aa: index for index, aa in enumerate("ACDE")}
    log_probs = np.array([
        [-1.0, -2.0, -3.0, -4.0],
        [-4.0, -1.0, -2.0, -3.0],
        [-3.0, -4.0, -1.0, -2.0],
    ])

    scores = mutation_log_odds("ACD", ["ACD", "CCD", "CCE"], log_probs, aa_to_col)

    np.testing.assert_allclose(scores, [0.0, -1.0, -2.0])
