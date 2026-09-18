"""
Correction-retention measurement behind ERRATA E1.

Generates retention.json: stable-fact correction retention (k=1, quarantined)
and store size, for the policies ERRATA compares, at two decay constants.
Committed so retention.json is reproducible rather than an orphan data file
(the missing generator was itself one of this project's errata, E5).
"""
import json
import numpy as np
import memory_sim as m

SEEDS = list(range(12)) + list(range(200, 212))   # two disjoint sets, pooled
POLICIES = ["append_only", "hard_overwrite", "reconsolidation_only",
            "full_suppress_only", "full_boost_only"]
TAUS = [400, 6400]

if __name__ == "__main__":
    out = {}
    for p in POLICIES:
        for tau in TAUS:
            runs = [m.run(p, s, tau_flat=float(tau)) for s in SEEDS]
            out[f"{p}@{tau}"] = {
                "stable": float(np.nanmean([r[1][1]["stable"] for r in runs])),
                "size": float(np.mean([r[2] for r in runs])),
            }
            print(f"{p}@{tau}: retention={out[f'{p}@{tau}']['stable']:.3f} "
                  f"size={out[f'{p}@{tau}']['size']:.0f}")
    json.dump(out, open("retention.json", "w"), indent=2)
    print("\nsaved retention.json")
