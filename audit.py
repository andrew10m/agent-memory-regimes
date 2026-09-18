"""
Automated claim audit.

Every claim this project makes is restated here as an executable test and
checked on two disjoint seed sets, across the tau spectrum where a decay
constant is involved. A claim PASSES only if it holds on both seed sets.

This file exists because the project's errors were found one at a time by
hand, and each hand-check missed the next one. The protocol is now uniform:
same tau semantics for every policy (see memory_sim.TAU_GEOMEAN), same seeds,
same metric, no per-policy tuning.
"""
import numpy as np
import memory_sim as m

SEEDS = {"A": list(range(12)), "B": list(range(200, 212))}
TAUS = [25, 100, 400, 1600, 6400]


def measure(policy, seeds, tau):
    out = [m.run(policy, s, tau_flat=float(tau)) for s in seeds]
    return {"acc": float(np.mean([o[0] for o in out])),
            "stable": float(np.nanmean([o[1][1]["stable"] for o in out])),
            "volatile": float(np.nanmean([o[1][1]["volatile"] for o in out]))}


def report(name, results):
    ok = all(results.values())
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for k, v in results.items():
        if not v:
            print(f"         failed on: {k}")
    return ok


if __name__ == "__main__":
    cache = {(p, s, t): measure(p, SEEDS[s], t)
             for p in ["append_only", "flat_ttl", "importance", "hard_overwrite",
                       "reconsolidation_only", "type_aware",
                       "full_suppress_only", "full_boost_only"]
             for s in SEEDS for t in TAUS}
    G = lambda p, s, t, f="acc": cache[(p, s, t)][f]
    passed = []

    passed.append(report(
        "R1  append-only collapses at high tau; active removal does not (>15pp)",
        {s: G("hard_overwrite", s, 6400) - G("append_only", s, 6400) > 0.15 for s in SEEDS}))

    passed.append(report(
        "R1b append-only is FINE at low tau (gap < 2pp)",
        {s: abs(G("hard_overwrite", s, 25) - G("append_only", s, 25)) < 0.02 for s in SEEDS}))

    passed.append(report(
        "R2  plain deletion beats graded suppression at high tau (>3pp)",
        {s: G("hard_overwrite", s, 6400) - G("reconsolidation_only", s, 6400) > 0.03 for s in SEEDS}))

    passed.append(report(
        "R2b graded suppression still beats append-only at high tau (>10pp)",
        {s: G("reconsolidation_only", s, 6400) - G("append_only", s, 6400) > 0.10 for s in SEEDS}))

    passed.append(report(
        "R3  suppression beats boosting on retention, at every tau",
        {f"{s}/tau{t}": G("full_suppress_only", s, t, "stable") >
                        G("full_boost_only", s, t, "stable")
         for s in SEEDS for t in TAUS}))

    passed.append(report(
        "R4  confidence paradox: stable facts hold corrections worse, every tau",
        {f"{s}/tau{t}": G("append_only", s, t, "stable") <
                        G("append_only", s, t, "volatile")
         for s in SEEDS for t in TAUS}))

    passed.append(report(
        "R5  flat TTL is indistinguishable from append-only, every tau (<2pp)",
        {f"{s}/tau{t}": abs(G("flat_ttl", s, t) - G("append_only", s, t)) < 0.02
         for s in SEEDS for t in TAUS}))

    passed.append(report(
        "W2  RETRACTED claim 'type-conditional decay is harmful' -- "
        "must FAIL at matched tau to stay retracted",
        {f"{s}/tau{t}": G("type_aware", s, t) < G("append_only", s, t)
         for s in SEEDS for t in TAUS}))

    print("\n--- type-conditional decay at matched tau (the W2 retraction) ---")
    for t in TAUS:
        a = np.mean([G("append_only", s, t) for s in SEEDS])
        y = np.mean([G("type_aware", s, t) for s in SEEDS])
        print(f"  tau={t:>5}: append-only {a:.3f}  typed {y:.3f}  "
              f"{'typed better' if y > a else 'append better'}")

    print(f"\n{sum(passed[:7])}/7 robust claims passed. "
          f"W2 shows as {'FAIL' if not passed[7] else 'PASS'} "
          f"({'correctly retracted' if not passed[7] else 'UNEXPECTED -- revisit'}).")
