from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Match:
    gold_index: int
    pred_index: int
    score: float


def rouge_l_f1_matrix(golds: Sequence[str], preds: Sequence[str]) -> List[List[float]]:
    from rouge_score import rouge_scorer  # type: ignore

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    m: List[List[float]] = []
    for g in golds:
        row: List[float] = []
        for p in preds:
            s = scorer.score(g, p)["rougeL"].fmeasure
            row.append(float(s))
        m.append(row)
    return m


def hungarian_match(
    sim: List[List[float]], min_score: float
) -> Tuple[List[Match], List[int], List[int]]:
    """
    Perform one-to-one matching maximizing similarity.

    Returns: (matches, unmatched_gold_indices, unmatched_pred_indices)
    """
    if not sim:
        return ([], [], list(range(len(sim[0])))) if sim else ([], [], [])
    if not sim[0]:
        return ([], list(range(len(sim))), [])

    import numpy as np  # type: ignore
    from scipy.optimize import linear_sum_assignment  # type: ignore

    # Convert to cost matrix for minimization: cost = 1 - sim
    sim_arr = np.array(sim, dtype=float)
    cost = 1.0 - sim_arr
    gold_idx, pred_idx = linear_sum_assignment(cost)

    matches: List[Match] = []
    matched_g = set()
    matched_p = set()
    for gi, pi in zip(gold_idx.tolist(), pred_idx.tolist()):
        score = float(sim_arr[gi, pi])
        if score >= min_score:
            matches.append(Match(gold_index=gi, pred_index=pi, score=score))
            matched_g.add(gi)
            matched_p.add(pi)

    unmatched_g = [i for i in range(sim_arr.shape[0]) if i not in matched_g]
    unmatched_p = [i for i in range(sim_arr.shape[1]) if i not in matched_p]
    return matches, unmatched_g, unmatched_p

