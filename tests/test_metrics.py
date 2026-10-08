import math

from jobagent.evals.metrics import ndcg_at_k


def test_ndcg_perfect_and_worse():
    allg = [2, 1, 0, 0]
    assert ndcg_at_k([2, 1, 0, 0], allg, 4) == 1.0
    assert ndcg_at_k([0, 1, 2, 0], allg, 4) < 1.0
    assert ndcg_at_k([0, 0], [0, 0], 2) == 0.0
    assert math.isclose(ndcg_at_k([1], [2, 1], 1), 1 / 3)
