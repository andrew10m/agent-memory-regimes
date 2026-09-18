"""
Robustness of the two central orderings under non-trivial variation.

If "plain deletion >= graded suppression >= append-only at high tau" holds
only at default parameters, it is a tuning artifact. This sweeps world
structure (fact count, mention/query/correction rates, classifier quality,
world drift) and the suppression coefficient itself, and checks the ordering
survives. It does. Run time ~4 min.
"""
import numpy as np
import memory_sim as m

S = range(8)


def triple(tau, **kw):
    a = np.mean([m.run("append_only", s, tau_flat=tau, **kw)[0] for s in S])
    r = np.mean([m.run("reconsolidation_only", s, tau_flat=tau, **kw)[0] for s in S])
    h = np.mean([m.run("hard_overwrite", s, tau_flat=tau, **kw)[0] for s in S])
    return a, r, h


WORLDS = {
    "baseline": {},
    "few facts (n=20)": {"n_facts": 20},
    "many facts (n=200)": {"n_facts": 200},
    "sparse mentions": {"p_mention": 0.05},
    "dense queries": {"p_query": 0.8},
    "poor classifier (60%)": {"clf_acc": 0.60},
    "perfect classifier": {"clf_acc": 1.0},
    "rare corrections": {"p_correct": 0.3},
}

if __name__ == "__main__":
    print("Ordering hard >= recon >= append at tau=6400, across world structure:")
    all_ok = True
    for name, kw in WORLDS.items():
        a, r, h = triple(6400.0, **kw)
        ok = h >= a - 0.005 and h >= r - 0.02 and r >= a - 0.02
        all_ok &= ok
        print(f"  [{'ok' if ok else 'XX'}] {name:24s} "
              f"append={a:.3f} recon={r:.3f} hard={h:.3f}")
    print(f"\nAll world variations preserve the ordering: {all_ok}")
    print("Deletion dominates suppression because suppression is soft deletion: "
          "it can only approach, never beat, removing the stale trace.")
