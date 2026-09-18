import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("regimes.json")); TAUS = d["taus"]; D = d["data"]
STY = {"append_only": ("#b2182b", "-", "Append-only"),
       "type_aware": ("#5aae61", "-.", "Type-conditional decay"),
       "flat_ttl": ("#d6604d", "--", "Flat TTL (Copilot)"),
       "importance": ("#f4a582", "-.", "Importance-driven"),
       "reconsolidation_only": ("#92c5de", "-", "Reconsolidation (suppression)"),
       "hard_overwrite": ("#2166ac", "-", "Plain deletion")}

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

for p, (c, ls, lab) in STY.items():
    y = [D[p][str(t)]["acc"] for t in TAUS]
    e = [D[p][str(t)]["acc_std"] for t in TAUS]
    ax[0].errorbar(TAUS, y, yerr=e, fmt="o" + ls, color=c, lw=2, ms=4,
                   capsize=2, label=lab)
ax[0].set_xscale("log"); ax[0].set_xlabel("τ — decay constant (log)")
ax[0].set_ylabel("accuracy (24 seeds, two sets)")
ax[0].axvspan(20, 120, color="#2166ac", alpha=.07)
ax[0].axvspan(1000, 8000, color="#b2182b", alpha=.07)
ax[0].text(45, .885, "recency decides", fontsize=8.5, color="#2166ac", ha="center")
ax[0].text(2600, .885, "accumulated strength decides", fontsize=8.5, color="#b2182b", ha="center")
ax[0].set_ylim(.50, .91)
ax[0].grid(alpha=.25); ax[0].legend(fontsize=8, frameon=False, loc="lower left")
ax[0].set_title("A. The defect exists in only one retrieval regime",
                fontsize=10.5, loc="left")

gap_ho = [D["hard_overwrite"][str(t)]["acc"] - D["append_only"][str(t)]["acc"] for t in TAUS]
gap_rc = [D["reconsolidation_only"][str(t)]["acc"] - D["append_only"][str(t)]["acc"] for t in TAUS]
ax[1].plot(TAUS, [100 * g for g in gap_ho], "o-", color="#2166ac", lw=2,
           label="Plain deletion − append-only")
ax[1].plot(TAUS, [100 * g for g in gap_rc], "s--", color="#92c5de", lw=2,
           label="Reconsolidation − append-only")
ax[1].axhline(0, color="k", lw=.8)
ax[1].axvline(25, color="#555", ls=":", lw=1.2)
ax[1].annotate("accuracy-tuning lands here —\nand the defect vanishes",
               xy=(25, 0), xytext=(90, 16), fontsize=8.5, color="#555",
               arrowprops=dict(arrowstyle="->", color="#555"))
ax[1].set_xscale("log"); ax[1].set_xlabel("τ — decay constant (log)")
ax[1].set_ylabel("accuracy gain, pp")
ax[1].grid(alpha=.25); ax[1].legend(fontsize=8.5, frameon=False, loc="upper left")
ax[1].set_title(f"B. Up to +{100*max(gap_ho):.1f} pp — but only on the right",
                fontsize=10.5, loc="left")

fig.suptitle("Figure 1 — The defect lives in one retrieval regime; accuracy-tuning selects the other",
             fontsize=12, y=1.04, x=.01, ha="left")
fig.tight_layout(); fig.savefig("fig1_regimes.png", dpi=160, bbox_inches="tight")
print(f"ok | max gap hard_overwrite {100*max(gap_ho):+.1f} pp, reconsolidation {100*max(gap_rc):+.1f} pp")
