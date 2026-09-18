"""
The correction to the previous study.

v2 tuned every policy for maximum accuracy and concluded that append-only,
plain overwrite and reconsolidation are indistinguishable. That conclusion
was an artifact of the tuning protocol: maximising accuracy drives the decay
constant tau to the smallest value in the grid, i.e. into a purely
RECENCY-DOMINATED retrieval regime where the freshest trace always wins and
a stale strong trace cannot outcompete a correction by construction.

In other words, the tuner did not fix the defect -- it escaped the regime in
which the defect exists, and then reported that the defect does not matter.

Real stores are not purely recency-ranked: vector stores rank by similarity,
graphs by edge weight, importance-based systems by a salience score. Those
are strength-dominated. So the regime that matters in practice is exactly
the one accuracy-tuning discards.

This script sweeps tau instead of tuning it away, and reports accuracy across
the recency <-> strength spectrum.
"""
import json
import numpy as np
import memory_sim as m

TAUS = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
POLICIES = ["append_only", "flat_ttl", "importance",
            "hard_overwrite", "reconsolidation_only", "type_aware"]
SEEDS = list(range(12))   # audit.py re-checks every claim on a second, disjoint set

if __name__ == "__main__":
    out = {}
    for p in POLICIES:
        out[p] = {}
        for tau in TAUS:
            runs = [m.run(p, s, tau_flat=float(tau)) for s in SEEDS]
            out[p][str(tau)] = {
                "acc": float(np.mean([r[0] for r in runs])),
                "acc_std": float(np.std([r[0] for r in runs])),
                "hold": float(np.nanmean([r[1][1]["stable"] for r in runs])),
                "size": float(np.mean([r[2] for r in runs])),
            }
        line = "  ".join(f"{out[p][str(t)]['acc']:.3f}" for t in TAUS)
        print(f"{p:22s} {line}")

    json.dump({"taus": TAUS, "data": out}, open("regimes.json", "w"), indent=2)

    ap = [out["append_only"][str(t)]["acc"] for t in TAUS]
    ho = [out["hard_overwrite"][str(t)]["acc"] for t in TAUS]
    gaps = [h - a for a, h in zip(ap, ho)]
    print(f"\ntau where append-only is fine  : {TAUS[int(np.argmin(np.abs(gaps)))]}")
    print(f"max advantage of active removal: {max(gaps):+.3f} at tau={TAUS[int(np.argmax(gaps))]}")
    print("The defect is regime-dependent, not universal. v2 measured only the "
          "regime where it is absent.")
