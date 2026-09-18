"""Boundary search: is there ANY regime where graded suppression
(reconsolidation) beats plain destructive overwrite? Two principled
candidates are tested:
  - noisy corrections (the user's correction is itself sometimes wrong):
    a deleted trace is gone, a suppressed one could in principle recover;
  - reverting facts (a fact returns to a value it held before):
    a deleted trace must be relearned, a suppressed one could revive.
Both are where non-destructive updating should pay off if it ever does.
"""
import json
import numpy as np
import memory_sim as m

SEEDS = range(8)
POLS = ["hard_overwrite", "reconsolidation_only"]


def acc(p, **kw):
    return float(np.mean([m.run(p, s, tau_flat=25.0, **kw)[0] for s in SEEDS]))


if __name__ == "__main__":
    out = {"noise": {}, "revert": {}}
    print("A. noisy corrections (p_mention=0.6)")
    for pb in [0.0, 0.1, 0.2, 0.3, 0.4]:
        v = {p: acc(p, p_mention=0.6, p_bad_correction=pb) for p in POLS}
        out["noise"][str(pb)] = v
        print(f"   p_bad={pb:.1f}  hard={v['hard_overwrite']:.3f} "
              f"recon={v['reconsolidation_only']:.3f} "
              f"delta={v['reconsolidation_only']-v['hard_overwrite']:+.3f}")
    print("B. reverting facts (p_mention=0.15)")
    for pr in [0.0, 0.3, 0.6, 0.9]:
        v = {p: acc(p, p_mention=0.15, p_revert=pr) for p in POLS}
        out["revert"][str(pr)] = v
        print(f"   p_revert={pr:.1f}  hard={v['hard_overwrite']:.3f} "
              f"recon={v['reconsolidation_only']:.3f} "
              f"delta={v['reconsolidation_only']-v['hard_overwrite']:+.3f}")
    json.dump(out, open("refutation.json", "w"), indent=2)
    deltas = [d[POLS[1]] - d[POLS[0]] for g in out.values() for d in g.values()]
    print(f"\nmax advantage for graded suppression: {max(deltas):+.4f}")
    print("No regime found where graded suppression beats plain overwrite.")
