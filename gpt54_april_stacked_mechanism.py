from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs_latest")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

rows = []

for path in RUNS_DIR.glob("2026-04-*.json"):
    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)

    if r.get("model_name") != "openai/gpt-5.4":
        continue

    cond = r.get("condition", {})
    turns = r.get("turns", [])

    conf_by_turn = {
        int(t["turn_index"]): float(t["parsed_confidence"])
        for t in turns
        if t.get("turn_index") is not None and t.get("parsed_confidence") is not None
    }

    if 2 not in conf_by_turn:
        continue

    rows.append({
        "agent_type": cond.get("agent_type"),
        "reassurance_type": cond.get("reassurance_type"),
        "condition": f"{cond.get('agent_type')} / {cond.get('reassurance_type')}",
        "confidence_t2": conf_by_turn[2],
        "toc": r.get("toc_turn"),
        "accuracy": r.get("accuracy"),
    })

df = pd.DataFrame(rows).dropna()
df["toc"] = df["toc"].astype(float)
df["accuracy"] = df["accuracy"].astype(float)

order = [
    "jtc / calibrated",
    "jtc / miscalibrated",
    "nonjtc / calibrated",
    "nonjtc / miscalibrated"
]

summary = (
    df.groupby("condition")
    .agg(
        confidence_t2_mean=("confidence_t2", "mean"),
        confidence_t2_sem=("confidence_t2", lambda x: x.std(ddof=1) / (len(x) ** 0.5)),
        toc_mean=("toc", "mean"),
        toc_sem=("toc", lambda x: x.std(ddof=1) / (len(x) ** 0.5)),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sem=("accuracy", lambda x: x.std(ddof=1) / (len(x) ** 0.5)),
        n=("accuracy", "count")
    )
    .reindex(order)
    .reset_index()
)

summary.to_csv(OUT_DIR / "gpt54_april_stacked_mechanism_summary.csv", index=False)

x = range(len(order))
labels = ["JTC\nCal", "JTC\nMiscal", "NonJTC\nCal", "NonJTC\nMiscal"]

fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

# Panel 1: early confidence
axes[0].errorbar(
    x,
    summary["confidence_t2_mean"],
    yerr=summary["confidence_t2_sem"],
    marker="o",
    linewidth=2,
    capsize=4
)
axes[0].set_ylabel("Confidence at turn 2")
axes[0].set_ylim(0, 100)
axes[0].set_title("GPT-5.4: Early certainty, commitment timing, and accuracy")

# Panel 2: TOC
axes[1].errorbar(
    x,
    summary["toc_mean"],
    yerr=summary["toc_sem"],
    marker="o",
    linewidth=2,
    capsize=4
)
axes[1].set_ylabel("Turn of commitment")
axes[1].set_ylim(1.8, 5.2)
axes[1].axhline(2, linestyle="--", linewidth=1, alpha=0.5)
axes[1].text(-0.35, 2.05, "earliest commitment", fontsize=9, alpha=0.7)

# Panel 3: accuracy
axes[2].errorbar(
    x,
    summary["accuracy_mean"],
    yerr=summary["accuracy_sem"],
    marker="o",
    linewidth=2,
    capsize=4
)
axes[2].set_ylabel("Accuracy")
axes[2].set_ylim(0, 1.05)
axes[2].set_xticks(list(x))
axes[2].set_xticklabels(labels)
axes[2].set_xlabel("Condition")

# n labels
for i, n in enumerate(summary["n"]):
    axes[2].text(i, -0.13, f"n={int(n)}", ha="center", fontsize=9)

plt.tight_layout()

out = OUT_DIR / "gpt54_april_stacked_mechanism.png"
plt.savefig(out, dpi=300)
plt.close()

print("Saved:", out)
print("Saved summary:", OUT_DIR / "gpt54_april_stacked_mechanism_summary.csv")
