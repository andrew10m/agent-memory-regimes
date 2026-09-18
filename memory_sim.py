"""
Type-Conditional Decay + Reconsolidation for LLM-Agent Memory
=============================================================
Controlled simulation of a multi-trace agent memory store.

Two biologically motivated signals are tested:
  (1) type-conditional decay      <- systems consolidation / semanticisation
  (2) reconsolidation-on-conflict <- Nader et al. (2000) reconsolidation

STATUS: this is a SIMULATION, not benchmark validation. It shows that the
proposed mechanism is SUFFICIENT to reproduce (and remove) documented
failure modes under a standard strength x recency retrieval model. It does
not show that deployed systems fail for exactly this reason.

v2 changes after self-audit:
  - relapse test is now QUARANTINED. In v1 the fact kept being mentioned
    during the check window, so the metric measured how fast ordinary
    conversation repairs the store, not whether the correction held.
  - type-blind baselines are TUNED over their own decay/TTL constants and
    reported at their best setting, so the comparison is not rigged by
    giving only the proposed policy tuned hyper-parameters.
  - ablations separate the two halves of reconsolidation (suppress the
    stale trace vs boost the corrected one) and test reconsolidation
    with no type awareness at all.
"""

import json
import os
from multiprocessing import Pool
import numpy as np
from dataclasses import dataclass, field

TYPES = ["stable", "medium", "volatile"]
CHANGE_RATE = {"stable": 0.00010, "medium": 0.0015, "volatile": 0.006}
TAU_BY_TYPE = {"stable": 6000.0, "medium": 600.0, "volatile": 60.0}
TAU_GEOMEAN = 600.0   # (60*600*6000)**(1/3); see make_policy
CHECK_KS = (1, 3, 10)
CHECK_SCALE = 20


@dataclass
class Trace:
    value: int
    t_written: int
    strength: float = 1.0
    t_last_used: int = 0


@dataclass
class Store:
    traces: dict = field(default_factory=dict)
    conflicts: dict = field(default_factory=dict)


def retrieve(store, fid, t, tau_fn):
    """score = strength * exp(-age / tau); the winner answers.

    Every vector / graph store approximates this. The relapse mechanism
    falls straight out: a long-reinforced stale trace can outscore a
    freshly appended correction of strength 1.
    """
    tr = store.traces.get(fid, [])
    if not tr:
        return None
    best, best_s = None, -1.0
    for x in tr:
        s = x.strength * np.exp(-(t - x.t_written) / tau_fn(fid))
        if s > best_s:
            best_s, best = s, x
    return best


def make_policy(name, ptype, tau_flat, ttl_flat, imp_thresh):
    type_blind = name in ("append_only", "flat_ttl", "importance",
                          "reconsolidation_only", "hard_overwrite")
    # tau means the same thing for every policy: the geometric mean of its
    # effective decay constants. TAU_BY_TYPE has geomean 600, so a type-aware
    # policy at tau=T uses TAU_BY_TYPE scaled by T/600. Without this, a
    # type-aware policy is silently compared at a different operating point
    # than a type-blind one -- the error that produced retraction W2.
    tau_fn = ((lambda fid: tau_flat) if type_blind
              else (lambda fid: TAU_BY_TYPE[ptype[fid]] * (tau_flat / TAU_GEOMEAN)))

    does_reconsolidate = name in ("reconsolidation_only", "full",
                                  "full_suppress_only", "full_boost_only")
    does_hard_overwrite = name == "hard_overwrite"
    does_suppress = name != "full_boost_only"
    does_boost = name != "full_suppress_only"

    def on_write(store, fid, value, t):
        tr = store.traces.setdefault(fid, [])
        for x in tr:
            if x.value == value:
                x.strength += 1.0          # repetition effect, emergent
                x.t_written = t
                return
        tr.append(Trace(value, t, 1.0, t))

    def on_correction(store, fid, true_value, t):
        store.conflicts[fid] = store.conflicts.get(fid, 0) + 1
        tr = store.traces.setdefault(fid, [])

        if does_hard_overwrite:
            # The strongest obvious baseline, and what an UPDATE-style memory
            # operation (Mem0) approximates: throw the old traces away and
            # write the asserted value. Destructive and irreversible.
            store.traces[fid] = [Trace(true_value, t, 1.0, t)]
            return

        if not does_reconsolidate:
            on_write(store, fid, true_value, t)   # append-style systems
            return

        # Reconsolidation: reactivating a consolidated trace returns it to a
        # labile state; only then can it be modified. A correction that never
        # reactivates the stale trace cannot overwrite it.
        if does_suppress:
            for x in tr:
                if x.value != true_value:
                    x.strength *= 0.25
        boost = (1.0 + np.log1p(store.conflicts[fid])) if does_boost else 1.0
        hit = False
        for x in tr:
            if x.value == true_value:
                x.strength = max(x.strength, 1.0) + boost
                x.t_written = t
                hit = True
        if not hit:
            tr.append(Trace(true_value, t, 1.0 + boost, t))

    def on_tick(store, t, importance):
        if name in ("append_only", "reconsolidation_only", "hard_overwrite"):
            return
        for fid, tr in store.traces.items():
            keep = []
            for x in tr:
                if name == "flat_ttl":
                    if (t - x.t_last_used < ttl_flat
                            or t - x.t_written < ttl_flat):
                        keep.append(x)
                elif name == "importance":
                    if importance[fid] > imp_thresh or t - x.t_written < 200:
                        keep.append(x)
                else:
                    if x.strength * np.exp(-(t - x.t_written) / tau_fn(fid)) > 0.02:
                        keep.append(x)
            store.traces[fid] = keep

    return tau_fn, on_write, on_correction, on_tick


def run(name, seed, tau_flat=600.0, ttl_flat=800.0, imp_thresh=0.35,
        n_facts=60, T=8000, p_mention=0.60, p_query=0.40,
        p_correct=0.85, clf_acc=0.85, p_bad_correction=0.0, p_revert=0.0):
    rng = np.random.default_rng(seed)

    ftype = [TYPES[i % 3] for i in range(n_facts)]
    rng.shuffle(ftype)
    # noisy type classifier (a small embedding head), NOT an oracle
    ptype = [tt if rng.random() < clf_acc else TYPES[rng.integers(0, 3)]
             for tt in ftype]
    importance = rng.random(n_facts)   # salience prior, independent of volatility

    truth = np.zeros(n_facts, dtype=int)
    store = Store()
    tau_fn, on_write, on_correction, on_tick = make_policy(
        name, ptype, tau_flat, ttl_flat, imp_thresh)
    for i in range(n_facts):
        on_write(store, i, 0, 0)

    correct_log, size_log, pending = [], [], []
    quarantine = {}       # fid -> t_until: no traffic reaches this fact
    hold = {k: {tt: [0, 0] for tt in TYPES} for k in CHECK_KS}
    MAXH = max(CHECK_KS) * CHECK_SCALE

    for t in range(1, T):
        for i in range(n_facts):
            if rng.random() < CHANGE_RATE[ftype[i]]:
                if truth[i] > 0 and rng.random() < p_revert:
                    truth[i] -= 1          # fact reverts to a previous value
                else:
                    truth[i] += 1

        if rng.random() < p_mention:
            i = int(rng.integers(0, n_facts))
            if quarantine.get(i, -1) < t:
                on_write(store, i, int(truth[i]), t)

        if rng.random() < p_query:
            i = int(rng.integers(0, n_facts))
            if quarantine.get(i, -1) < t:
                tr = retrieve(store, i, t, tau_fn)
                ok = tr is not None and tr.value == truth[i]
                correct_log.append(1 if ok else 0)
                if tr is not None:
                    tr.t_last_used = t
                if not ok and rng.random() < p_correct:
                    # A correction is not always right: users misremember,
                    # upstream extraction misfires. The asserted value may be
                    # wrong, but retention is always scored against the TRUTH.
                    if rng.random() < p_bad_correction:
                        asserted = int(truth[i]) + int(rng.integers(1, 5))
                    else:
                        asserted = int(truth[i])
                    on_correction(store, i, asserted, t)
                    quarantine[i] = t + MAXH
                    for k in CHECK_KS:
                        pending.append((t + k * CHECK_SCALE, i,
                                        int(truth[i]), k, ftype[i]))

        still = []
        for (t_chk, i, val, k, tt) in pending:
            if t >= t_chk:
                if truth[i] != val:
                    continue                       # world moved on: invalid test
                tr = retrieve(store, i, t, tau_fn)  # read-only probe
                hold[k][tt][1] += 1
                if tr is not None and tr.value == val:
                    hold[k][tt][0] += 1
            else:
                still.append((t_chk, i, val, k, tt))
        pending = still

        if t % 50 == 0:
            on_tick(store, t, importance)
            size_log.append(sum(len(v) for v in store.traces.values()))

    acc = float(np.mean(correct_log[len(correct_log) // 5:]))
    h = {k: {tt: (hold[k][tt][0] / hold[k][tt][1] if hold[k][tt][1] else np.nan)
             for tt in TYPES} for k in CHECK_KS}
    return acc, h, float(np.mean(size_log[-20:]))


POLICIES = ["append_only", "flat_ttl", "importance", "hard_overwrite",
            "type_aware", "reconsolidation_only",
            "full_suppress_only", "full_boost_only", "full"]

LABELS = {
    "append_only": "Append-only (vector store)",
    "flat_ttl": "Flat TTL (Copilot-style)",
    "importance": "Importance-driven (TiM/MemTool)",
    "hard_overwrite": "Hard overwrite on conflict (Mem0 UPDATE-like)",
    "type_aware": "Type-conditional decay only",
    "reconsolidation_only": "Reconsolidation only (type-blind)",
    "full_suppress_only": "Ablation: suppress only",
    "full_boost_only": "Ablation: boost only",
    "full": "Type-decay + reconsolidation (full)",
}

TAU_GRID = [25.0, 50.0, 100.0, 150.0, 300.0, 600.0, 1200.0, 2400.0]
TTL_GRID = [100.0, 200.0, 400.0, 800.0, 1600.0]
IMP_GRID = [0.2, 0.35, 0.5, 0.7]   # 0.0 removed: it makes the policy
                                   # forget nothing, collapsing it onto append_only
TYPE_MULT_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]

# Hyper-parameters are chosen on TUNE_SEEDS and reported on EVAL_SEEDS.
# The two sets are disjoint: tuning and reporting on the same seeds would be
# fitting the test set, and the whole point of this script is a fair
# comparison between policies.
TUNE_SEEDS = list(range(4))
EVAL_SEEDS = list(range(4, 16))


def _one(args):
    name, seed, tau, ttl, imp = args
    return run(name, seed, tau_flat=tau, ttl_flat=ttl, imp_thresh=imp)


def evaluate(name, tau, ttl, imp, seeds, pool=None):
    jobs = [(name, s, tau, ttl, imp) for s in seeds]
    out = pool.map(_one, jobs) if pool else [_one(j) for j in jobs]
    return (float(np.mean([o[0] for o in out])),
            float(np.std([o[0] for o in out])),
            [o[1] for o in out],
            float(np.mean([o[2] for o in out])))


def grid_for(policy):
    if policy == "importance":
        return [(tau, 800.0, th) for tau in TAU_GRID for th in IMP_GRID]
    if policy == "flat_ttl":
        return [(tau, ttl, 0.35) for tau in TAU_GRID for ttl in TTL_GRID]
    if policy in ("append_only", "reconsolidation_only", "hard_overwrite"):
        return [(tau, 800.0, 0.35) for tau in TAU_GRID]
    # type-aware policies: tune a global multiplier on TAU_BY_TYPE
    return [(mul, 800.0, 0.35) for mul in TYPE_MULT_GRID]


if __name__ == "__main__":
    import sys
    if "--quick" in sys.argv:
        TAU_GRID = [25.0, 100.0, 600.0, 2400.0]
        TTL_GRID = [200.0, 800.0]
        IMP_GRID = [0.0, 0.35]
        TUNE_SEEDS, EVAL_SEEDS = list(range(2)), list(range(2, 7))
        print("[--quick] coarse grid, 5 eval seeds. README numbers use the "
              "full grid and 12 eval seeds.\n")

    pool = Pool(os.cpu_count()) if os.cpu_count() > 1 else None
    results = {}
    for p in POLICIES:
        best_cfg, best_m = None, -np.inf
        for cfg in grid_for(p):
            m, _, _, _ = evaluate(p, *cfg, TUNE_SEEDS, pool)
            if m > best_m:
                best_m, best_cfg = m, cfg
        m, sd, hs, sz = evaluate(p, *best_cfg, EVAL_SEEDS, pool)
        results[p] = {
            "acc_mean": m, "acc_std": sd, "size": sz,
            "tuned": {"tau": best_cfg[0], "ttl": best_cfg[1], "imp": best_cfg[2]},
            "hold": {str(k): {tt: float(np.nanmean([h[k][tt] for h in hs]))
                              for tt in TYPES} for k in CHECK_KS},
            "hold_std": {str(k): {tt: float(np.nanstd([h[k][tt] for h in hs]))
                                  for tt in TYPES} for k in CHECK_KS},
            "hold_raw": [[[float(h[k][tt]) for tt in TYPES] for k in CHECK_KS]
                         for h in hs],
        }
        print(f"{LABELS[p]:38s} acc={m:.3f}+/-{sd:.3f} traces={sz:5.0f} "
              f"hold@1_stable={results[p]['hold']['1']['stable']:.3f}")
    if pool:
        pool.close()
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results.json")
