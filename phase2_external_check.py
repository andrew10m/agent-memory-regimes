"""
Phase 2 -- external check against real MemoryAgentBench evaluation.

This script does NOT run a new experiment on real data (no HuggingFace
access, no LLM API key in this environment -- see README Phase 2 section
for why). It records a third-party, real-model result that is publicly
posted (HUST-AI-HYZ/MemoryAgentBench issue #18: "Mnemos" typed conflict
resolution vs naive baseline, GPT-4.1-mini, full Conflict_Resolution
split, 800 questions) and checks it against the qualitative prediction
made by the simulation in this repo: that resolving conflicts by TYPE
alone, without a mechanism to avoid suppressing non-conflicting traces,
degrades as scale/complexity grows.

Source: https://github.com/HUST-AI-HYZ/MemoryAgentBench/issues/18
Numbers transcribed verbatim from the issue; not independently reproduced.
"""
import json

# verbatim from the issue
MNEMOS = {
    "FC-MH": {"6K": (27.0, 9.0), "32K": (11.0, 3.0), "64K": (8.0, 6.0), "262K": (2.0, 2.0)},
    "FC-SH": {"6K": (90.0, 69.0), "32K": (65.0, 80.0), "64K": (55.0, 76.0), "262K": (28.0, 76.0)},
}

if __name__ == "__main__":
    print("Real-model check (Mnemos vs naive, GPT-4.1-mini, full CR split, n=800)")
    print(f"{'split':10}{'typed':>8}{'naive':>8}{'delta':>8}")
    for split, d in MNEMOS.items():
        for ctx, (typed, naive) in d.items():
            print(f"{split}-{ctx:<5}{typed:8.1f}{naive:8.1f}{typed - naive:+8.1f}")

    mh_typed = sum(v[0] for v in MNEMOS["FC-MH"].values()) / 4
    mh_naive = sum(v[1] for v in MNEMOS["FC-MH"].values()) / 4
    print(f"\nFC-MH average: typed {mh_typed:.1f}% vs naive {mh_naive:.1f}%  "
          f"-> typed conflict handling helps multi-hop")

    sh_long = [MNEMOS["FC-SH"][k] for k in ("32K", "64K", "262K")]
    wins = sum(1 for t, n in sh_long if t < n)
    print(f"FC-SH at 32K/64K/262K: typed loses to naive in {wins}/3 cases  "
          f"-> typed handling HURTS long single-hop")
    print("\nAuthor's own diagnosis (issue text): 'over-deletes similar-but-"
          "non-contradictory facts' at long context.")
    print("\nThis is the same qualitative failure as the type_aware ablation in "
          "this repo's simulation: classifying/typing conflicts without a "
          "mechanism to protect non-conflicting traces suppresses things it "
          "should not. The simulation predicted the shape of this failure; "
          "it did not predict these numbers and was not tuned to match them.")

    json.dump(MNEMOS, open("mnemos_external.json", "w"), indent=2)
