from __future__ import annotations

from types import SimpleNamespace

import torch

from prospero.search.prosst_puct import AA20, decode_with_puct


class FakeTokenizer:
    mask_token_id = 999

    def __call__(self, seqs, return_tensors="pt", padding=False):
        rows = []
        for seq in seqs:
            rows.append([100, *[AA20.index(aa) for aa in seq], 101])
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class FakeProSSTGenerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = FakeTokenizer()
        self.aa_to_id = {aa: idx for idx, aa in enumerate(AA20)}
        self.id_to_aa = {idx: aa for idx, aa in enumerate(AA20)}
        self.full_ids = torch.tensor(list(range(len(AA20))), dtype=torch.long)
        self.alphabet = {aa: AA20 for aa in AA20}
        self.args = SimpleNamespace(smc_vocab="cluster", non_cluster_logit_penalty=0.0)

    def _tokenize_batch(self, seqs):
        out = self.tokenizer(seqs, return_tensors="pt", padding=False)
        return out["input_ids"], out["attention_mask"]

    def _distribution_ids(self, original_aa):
        return torch.tensor([self.aa_to_id[aa] for aa in self.alphabet[original_aa]], dtype=torch.long)

    def logits_for_input_ids(self, input_ids, attention_mask):
        batch, tokens = input_ids.shape
        logits = torch.full((batch, tokens - 2, len(AA20)), -8.0)
        for row in range(batch):
            masked = torch.nonzero(input_ids[row] == self.tokenizer.mask_token_id, as_tuple=False).flatten()
            if masked.numel() == 0:
                continue
            pos = int(masked[0].item()) - 1
            logits[row, pos, AA20.index("A")] = 0.0
            logits[row, pos, AA20.index("C" if pos == 0 else "D")] = 5.0
        return logits


def test_prosst_puct_adapter_decodes_high_delta_sequence():
    generator = FakeProSSTGenerator()

    final_state, summary = decode_with_puct(
        generator,
        covered_start="AAA",
        mask_positions=[0, 1],
        simulations=40,
        c_puct=1.5,
    )

    assert "".join(final_state.sequence) == "CDA"
    assert final_state.score > 9.0
    assert len(final_state.steps) == 2
    assert summary["terminal_count"] > 0
    assert summary["terminal_score_max"] == final_state.score

