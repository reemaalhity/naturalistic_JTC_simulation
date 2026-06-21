from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs_latest")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

rows = []

for path in RUNS_DIR.glob("2026-04-*.json"):
    r = json.load(open(path, encoding="utf-8"))

    if r.get("model_name") != "openai/gpt-5.4":
        continue

    cond = r.get("condition", {})
    if str(cond.get("agent_type")).lower() != "jtc":
        continue

    conf_by_turn = {
        int(t["turn_index"]): float(t["parsed_confidence"])
        for t in r.get("turns", [])
        if t.get("turn_index") is not None and t.get("parsed_confidence") is not None
    }

    if 2 not in conf_by_turn:
        continue

    rows.append({
        "reassurance": cond.get("reassurance_type"),
        "confidence_t2": conf_by_turn[2],
        "toc": int(r.get("toc_turn")),
        "accuracy": int(r.get("accuracy")),
    })

df = pd.DataFrame(rows).dropna()

fig, ax = plt.subplots(figsize=(7, 5))

for reassurance, sub in df.groupby("reassurance"):
    correct = sub[sub["accuracy"] == 1]
    incorrect = sub[sub["accuracy"] == 0]

    ax.scatter(
        correct["confidence_t2"],
        correct["toc"],
        s=90,
        alpha=0.7,
        label=f"{reassurance} correct"
    )

    ax.scatter(
        incorrect["confidence_t2"],
        incorrect["toc"],
        s=150,
        marker="x",
        linewidths=2.8,
        label=f"{reassurance} incorrect"
    )

ax.axhline(2, linestyle="--", linewidth=1, alpha=0.5)
ax.text(df["confidence_t2"].min() - 3, 2.05, "earliest commitment", fontsize=9, alpha=0.7)

ax.set_xlabel("Early confidence at turn 2")
ax.set_ylabel("Turn of commitment")
ax.set_title("GPT-5.4 JTC runs: early confidence, commitment, and accuracy")

ax.set_ylim(1.8, 3.2)
ax.set_xlim(df["confidence_t2"].min() - 5, df["confidence_t2"].max() + 5)

ax.legend(frameon=True, fontsize=9)

plt.tight_layout()
out = OUT_DIR / "gpt54_april_jtc_clean_mechanism.png"
plt.savefig(out, dpi=300)
plt.close()

print("Saved:", out)
