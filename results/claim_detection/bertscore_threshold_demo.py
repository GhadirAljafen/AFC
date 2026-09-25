"""
Evidence: why a RAW BERTScore threshold of 0.85 is not a valid matching cutoff.

Scores two kinds of sentence pairs under both raw and baseline-rescaled
BERTScore (roberta-large, lang="en"):

  * UNRELATED  — different topics entirely; must NOT match.
  * PARAPHRASE — same fact, different wording; SHOULD match.

The point is the SEPARATION between those two groups. Raw BERTScore compresses
everything into a narrow high band, leaving almost no room between "unrelated"
and "paraphrase"; baseline rescaling re-centers unrelated pairs near 0 and opens
that gap up.

Run:  python results/claim_detection/bertscore_threshold_demo.py
Writes: bertscore_threshold_demo.json (next to this file)

Requires torch + transformers<5 (transformers>=5 needs torch>=2.4, which is
unavailable on Intel macOS — see BERTSCORE_ANALYSIS.md).
"""

import json
import os

from bert_score import score

RAW_THRESHOLD_UNDER_TEST = 0.85

UNRELATED = [
    ("Saudi Arabia is the world's largest oil producer",
     "The funeral was held on Tuesday morning"),
    ("Inflation hit 3.5 percent in January",
     "Six Canadian MPs were denied entry to the West Bank"),
    ("Acme Corporation announced a major expansion",
     "Miss Finland's photo sparked international debate"),
]
PARAPHRASE = [
    ("Saudi Arabia is the world's largest oil producer",
     "Saudi Arabia produces more oil than any other country"),
    ("Charges were dropped in March 2024",
     "The case was dismissed in March 2024"),
]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    pairs = [(a, b, "UNRELATED") for a, b in UNRELATED] + \
            [(a, b, "PARAPHRASE") for a, b in PARAPHRASE]
    cands = [a for a, _, _ in pairs]
    refs = [b for _, b, _ in pairs]

    _, _, raw = score(cands, refs, lang="en", rescale_with_baseline=False, verbose=False)
    _, _, res = score(cands, refs, lang="en", rescale_with_baseline=True, verbose=False)
    raw, res = [float(x) for x in raw], [float(x) for x in res]

    print("\n" + "=" * 80)
    print(f"{'RAW':>7} {'RESCALED':>9}  {'verdict @ raw 0.85':>20}   pair")
    print("=" * 80)
    rows = []
    for (a, b, kind), r, s in zip(pairs, raw, res):
        passes = r >= RAW_THRESHOLD_UNDER_TEST
        if kind == "UNRELATED":
            verdict = "FALSE ACCEPT" if passes else "correctly rejected"
        else:
            verdict = "correctly accepted" if passes else "FALSE REJECT"
        print(f"{r:7.3f} {s:9.3f}  {verdict:>20}   [{kind}]")
        print(f"{'':40}{a[:46]!r}")
        print(f"{'':40}vs {b[:46]!r}")
        rows.append({"candidate": a, "reference": b, "kind": kind,
                     "raw_f1": r, "rescaled_f1": s, "passes_raw_085": passes})

    u_raw = [r for r, (_, _, k) in zip(raw, pairs) if k == "UNRELATED"]
    p_raw = [r for r, (_, _, k) in zip(raw, pairs) if k == "PARAPHRASE"]
    u_res = [s for s, (_, _, k) in zip(res, pairs) if k == "UNRELATED"]
    p_res = [s for s, (_, _, k) in zip(res, pairs) if k == "PARAPHRASE"]

    raw_sep = mean(p_raw) - mean(u_raw)
    res_sep = mean(p_res) - mean(u_res)

    print("=" * 80)
    print(f"RAW      : unrelated={mean(u_raw):.3f}  paraphrase={mean(p_raw):.3f}"
          f"  -> separation={raw_sep:.3f}")
    print(f"RESCALED : unrelated={mean(u_res):.3f}  paraphrase={mean(p_res):.3f}"
          f"  -> separation={res_sep:.3f}")
    print(f"\nRescaling widens the usable margin by {res_sep / raw_sep:.1f}x.")
    print(f"Raw headroom between the unrelated floor ({max(u_raw):.3f}) and the "
          f"0.85 cutoff: {RAW_THRESHOLD_UNDER_TEST - max(u_raw):.3f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bertscore_threshold_demo.json")
    with open(out, "w") as f:
        json.dump({
            "model": "roberta-large",
            "raw_threshold_under_test": RAW_THRESHOLD_UNDER_TEST,
            "pairs": rows,
            "summary": {
                "raw": {"unrelated_mean": mean(u_raw), "paraphrase_mean": mean(p_raw),
                        "separation": raw_sep, "unrelated_max": max(u_raw)},
                "rescaled": {"unrelated_mean": mean(u_res), "paraphrase_mean": mean(p_res),
                             "separation": res_sep},
                "separation_gain": res_sep / raw_sep if raw_sep else None,
            },
        }, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
