from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.src.io import read_jsonl, write_jsonl
from eval.src.paths import get_paths
from eval.src.matching import hungarian_match, rouge_l_f1_matrix

# Prevent Matplotlib/font cache warnings in environments where home cache dirs are unwritable.
# bert-score imports matplotlib for optional plotting; we don't need it for scoring.
import os
os.environ.setdefault("MPLBACKEND", "Agg")


def score_split(split: str, limit: int | None = None) -> Path:
    """
    Placeholder scoring stage.

    Next steps (to be implemented):
      - load gold from normalized split
      - load predictions from intermediate
      - compute similarity + Hungarian matching
      - write per-article results + aggregates
    """
    paths = get_paths()
    norm_path = paths.intermediate_dir / f"newsscope_{split}_normalized.jsonl"
    pred_path = paths.intermediate_dir / f"predictions_{split}.jsonl"
    if not norm_path.exists() or not pred_path.exists():
        raise FileNotFoundError("Missing normalized dataset or predictions. Run ingest + run-agent first.")

    # Two-stage similarity:
    # - ROUGE-L used as cheap filter
    # - BERTScore F1 used for final matching where available
    min_rouge = 0.00
    min_bert_f1 = 0.65
    near_miss_min = 0.70

    gold_by_id: Dict[str, Dict[str, Any]] = {}
    for i, ex in enumerate(read_jsonl(norm_path)):
        if limit is not None and i >= limit:
            break
        eid = str(ex.get("example_id"))
        gold_claims = ex.get("gold_claims") if isinstance(ex.get("gold_claims"), list) else []
        gold_by_id[eid] = {
            "gold_claims": gold_claims,
            "domain": ex.get("domain"),
            "source": ex.get("source"),
        }

    rows_out: List[Dict[str, Any]] = []
    micro_tp = micro_fp = micro_fn = 0
    retrieval_slices: Dict[str, Dict[str, int]] = {}

    for i, pr in enumerate(read_jsonl(pred_path)):
        if limit is not None and i >= limit:
            break
        eid = str(pr.get("example_id"))
        pred_claims = pr.get("pred_claims") if isinstance(pr.get("pred_claims"), list) else []
        gold_claims = gold_by_id.get(eid, {}).get("gold_claims", [])

        gold_texts = [gc.get("claim_text", "") for gc in gold_claims if isinstance(gc, dict)]
        pred_texts = [pc.get("claim_text", "") for pc in pred_claims if isinstance(pc, dict)]

        rouge = rouge_l_f1_matrix(gold_texts, pred_texts) if gold_texts and pred_texts else []

        bert = _bertscore_f1_matrix_filtered(
            golds=gold_texts,
            preds=pred_texts,
            rouge_matrix=rouge,
            min_rouge=min_rouge,
            top_k_per_gold=12,
        )

        # Use BERTScore when we have it; otherwise fall back to ROUGE-L.
        sim = bert if bert else rouge

        matches, unmatched_g, unmatched_p = (
            hungarian_match(sim, min_score=min_bert_f1 if bert else 0.30)
            if sim
            else ([], list(range(len(gold_texts))), list(range(len(pred_texts))))
        )

        tp = len(matches)
        fp = len(unmatched_p)
        fn = len(unmatched_g)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        rs = pr.get("retrieval_source") or "unknown"
        retrieval_slices.setdefault(rs, {"tp": 0, "fp": 0, "fn": 0, "examples": 0})
        retrieval_slices[rs]["tp"] += tp
        retrieval_slices[rs]["fp"] += fp
        retrieval_slices[rs]["fn"] += fn
        retrieval_slices[rs]["examples"] += 1

        rows_out.append(
            {
                "example_id": eid,
                "split": split,
                "domain": gold_by_id.get(eid, {}).get("domain"),
                "retrieval_source": pr.get("retrieval_source"),
                "gold_count": len(gold_texts),
                "pred_count": len(pred_texts),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "matches": [
                    {
                        "gold_index": m.gold_index,
                        "pred_index": m.pred_index,
                        "score": m.score,
                        "score_kind": "bertscore_f1" if bert else "rougeL_f1",
                    }
                    for m in matches
                ],
                "unmatched_gold": unmatched_g,
                "unmatched_pred": unmatched_p,
            }
        )

    prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    out_path = paths.reports_dir / f"per_article_results_{split}.jsonl"
    write_jsonl(out_path, rows_out)

    summary_path = paths.reports_dir / f"metrics_summary_{split}.json"
    summary_path.write_text(
        __import__("json").dumps(
            {
                "split": split,
                "matching": {
                    "min_rouge_filter": min_rouge,
                    "min_bertscore_f1": min_bert_f1,
                    "near_miss_min": near_miss_min,
                },
                "micro": {"tp": micro_tp, "fp": micro_fp, "fn": micro_fn, "precision": prec, "recall": rec, "f1": f1},
                "by_retrieval_source": {
                    k: {
                        **v,
                        "precision": (v["tp"] / (v["tp"] + v["fp"])) if (v["tp"] + v["fp"]) else 0.0,
                        "recall": (v["tp"] / (v["tp"] + v["fn"])) if (v["tp"] + v["fn"]) else 0.0,
                    }
                    for k, v in retrieval_slices.items()
                },
                "note": "Scores use BERTScore F1 where installed; otherwise fall back to ROUGE-L.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_error_bundles(split=split, per_article_results=rows_out, gold_by_id=gold_by_id, pred_path=pred_path)
    return out_path


def _bertscore_f1_matrix_filtered(
    golds: List[str],
    preds: List[str],
    rouge_matrix: List[List[float]],
    min_rouge: float,
    top_k_per_gold: int,
) -> List[List[float]]:
    """
    Compute a sparse BERTScore F1 matrix, limited to likely candidate pairs selected by ROUGE-L.

    Returns [] if BERTScore isn't available.
    """
    try:
        from bert_score import score as bert_score  # type: ignore
    except Exception:
        return []

    if not golds or not preds:
        return []

    # Build candidate pairs (gi, pi) using ROUGE prefilter + top-k.
    candidates: List[tuple[int, int]] = []
    for gi, row in enumerate(rouge_matrix):
        scored = [(pi, s) for pi, s in enumerate(row) if s >= min_rouge]
        scored.sort(key=lambda x: x[1], reverse=True)
        for pi, _ in scored[:top_k_per_gold]:
            candidates.append((gi, pi))

    if not candidates:
        return []

    cand_g = [golds[gi] for gi, _ in candidates]
    cand_p = [preds[pi] for _, pi in candidates]

    try:
        # Requires PyTorch + compatible transformers; venv may lack them — fall back to ROUGE-L.
        P, R, F1 = bert_score(cand_p, cand_g, lang="en", rescale_with_baseline=True, verbose=False)
        f1_list = [float(x) for x in F1.tolist()]
    except Exception:
        return []

    mat = [[0.0 for _ in range(len(preds))] for _ in range(len(golds))]
    for (gi, pi), f1 in zip(candidates, f1_list):
        mat[gi][pi] = f1
    return mat


def _write_error_bundles(
    split: str,
    per_article_results: List[Dict[str, Any]],
    gold_by_id: Dict[str, Dict[str, Any]],
    pred_path: Path,
) -> None:
    paths = get_paths()

    # Load predictions for text lookup.
    preds_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for pr in read_jsonl(pred_path):
        eid = str(pr.get("example_id"))
        preds_by_id[eid] = pr.get("pred_claims") if isinstance(pr.get("pred_claims"), list) else []

    fp_rows: List[Dict[str, Any]] = []
    fn_rows: List[Dict[str, Any]] = []
    near_rows: List[Dict[str, Any]] = []

    for r in per_article_results:
        eid = str(r.get("example_id"))
        gold_claims = gold_by_id.get(eid, {}).get("gold_claims", [])
        gold_texts = [gc.get("claim_text", "") for gc in gold_claims if isinstance(gc, dict)]
        pred_claims = preds_by_id.get(eid, [])
        pred_texts = [pc.get("claim_text", "") for pc in pred_claims if isinstance(pc, dict)]

        unmatched_gold = r.get("unmatched_gold") or []
        unmatched_pred = r.get("unmatched_pred") or []

        for pi in unmatched_pred:
            if 0 <= int(pi) < len(pred_texts):
                fp_rows.append(
                    {
                        "example_id": eid,
                        "split": split,
                        "domain": r.get("domain"),
                        "retrieval_source": r.get("retrieval_source"),
                        "pred_index": int(pi),
                        "pred_claim": pred_texts[int(pi)],
                    }
                )

        for gi in unmatched_gold:
            if 0 <= int(gi) < len(gold_texts):
                fn_rows.append(
                    {
                        "example_id": eid,
                        "split": split,
                        "domain": r.get("domain"),
                        "retrieval_source": r.get("retrieval_source"),
                        "gold_index": int(gi),
                        "gold_claim": gold_texts[int(gi)],
                    }
                )

        # Near-miss heuristic: unmatched items with best ROUGE-L in [0.1, 0.3)
        if gold_texts and pred_texts:
            rouge = rouge_l_f1_matrix(gold_texts, pred_texts)
            for gi in unmatched_gold:
                gi = int(gi)
                if 0 <= gi < len(rouge):
                    best_pi, best = max(enumerate(rouge[gi]), key=lambda x: x[1])
                    if 0.10 <= best < 0.30:
                        near_rows.append(
                            {
                                "example_id": eid,
                                "split": split,
                                "domain": r.get("domain"),
                                "retrieval_source": r.get("retrieval_source"),
                                "gold_index": gi,
                                "pred_index": best_pi,
                                "gold_claim": gold_texts[gi],
                                "pred_claim": pred_texts[best_pi],
                                "rougeL_f1": float(best),
                            }
                        )

    write_jsonl(paths.reports_dir / f"false_positives_{split}.jsonl", fp_rows)
    write_jsonl(paths.reports_dir / f"missed_claims_{split}.jsonl", fn_rows)
    write_jsonl(paths.reports_dir / f"near_misses_{split}.jsonl", near_rows)

