from __future__ import annotations

from dataclasses import dataclass

from prospero.search.puct import ActionPrior, PUCTConfig, PUCTSearch


@dataclass(frozen=True)
class ToyState:
    prefix: tuple[str, ...] = ()
    score: float = 0.0
    depth: int = 0


class ToyEvaluator:
    def __init__(self, depth=2):
        self.depth = depth
        self.values = {"A": 0.2, "B": 1.0, "C": -0.4}
        self.priors = {"A": 0.85, "B": 0.1, "C": 0.05}

    def is_terminal(self, state):
        return state.depth >= self.depth

    def terminal_value(self, state):
        return state.score

    def actions(self, state):
        return [
            ActionPrior(action=action, prior=self.priors[action], immediate_value=self.values[action])
            for action in ["A", "B", "C"]
        ]

    def transition(self, state, action, immediate_value):
        return ToyState(
            prefix=(*state.prefix, action),
            score=state.score + immediate_value,
            depth=state.depth + 1,
        )

    def rollout_value(self, state):
        remaining = self.depth - state.depth
        return state.score + remaining * self.values["B"]


def test_puct_finds_terminal_states_and_ranks_by_terminal_value():
    search = PUCTSearch(ToyEvaluator(depth=2), PUCTConfig(simulations=200, c_puct=1.25))

    results = search.run(ToyState())

    assert results
    assert results[0].state.prefix == ("B", "B")
    assert results[0].value == 2.0
    assert all(result.state.depth == 2 for result in results)


def test_puct_backup_updates_root_and_edge_visits():
    search = PUCTSearch(ToyEvaluator(depth=1), PUCTConfig(simulations=20, c_puct=1.0))

    search.run(ToyState())

    assert search.root is not None
    assert search.root.visits == 20
    assert sum(edge.visits for edge in search.root.edges.values()) == 19
    assert any(edge.value_sum != 0.0 for edge in search.root.edges.values())


def test_puct_uses_priors_to_explore_promising_unvisited_edges():
    search = PUCTSearch(ToyEvaluator(depth=1), PUCTConfig(simulations=4, c_puct=5.0))

    search.run(ToyState())

    assert search.root is not None
    visits = {action: edge.visits for action, edge in search.root.edges.items()}
    assert visits["A"] >= visits["C"]


def test_puct_is_deterministic_for_fixed_evaluator():
    config = PUCTConfig(simulations=50, c_puct=1.5)

    first = PUCTSearch(ToyEvaluator(depth=2), config).run(ToyState())
    second = PUCTSearch(ToyEvaluator(depth=2), config).run(ToyState())

    assert [(item.state, item.value, item.visits) for item in first] == [
        (item.state, item.value, item.visits) for item in second
    ]
