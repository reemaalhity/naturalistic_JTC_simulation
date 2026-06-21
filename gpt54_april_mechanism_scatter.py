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
        "condition": f"{cond.get('agent_type')} / {cond.get('reassurance_type')}",
        "early_confidence": conf_by_turn[2],
        "toc": r.get("toc_turn"),
        "accuracy": r.get("accuracy"),
    })

df = pd.DataFrame(rows).dropna()
df["toc"] = df["toc"].astype(int)
df["accuracy"] = df["accuracy"].astype(int)

condition_order = [
    "jtc / calibrated",
    "jtc / miscalibrated",
    "nonjtc / calibrated",
    "nonjtc / miscalibrated"
]

fig, ax = plt.subplots(figsize=(8, 6))

for condition in condition_order:
    sub = df[df["condition"] == condition]

    correct = sub[sub["accuracy"] == 1]
    incorrect = sub[sub["accuracy"] == 0]

    ax.scatter(
        correct["early_confidence"],
        correct["toc"],
        s=90,
        alpha=0.75,
        label=f"{condition} correct"
    )

    ax.scatter(
        incorrect["early_confidence"],
        incorrect["toc"],
        s=130,
        alpha=0.95,
        marker="x",
        linewidths=2.5,
        label=f"{condition} incorrect"
    )

ax.set_xlabel("Early confidence at turn 2")
ax.set_ylabel("Turn of commitment")
ax.set_title("GPT-5.4: Early confidence, commitment timing, and accuracy")

ax.set_ylim(1.8, 5.2)
ax.set_xlim(df["early_confidence"].min() - 5, df["early_confidence"].max() + 5)

ax.axhline(2, linestyle="--", linewidth=1, alpha=0.5)
ax.text(
    df["early_confidence"].min() - 4,
    2.05,
    "earliest commitment",
    fontsize=9,
    alpha=0.7
)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8,
    frameon=True
)

plt.tight_layout()

out = OUT_DIR / "gpt54_april_mechanism_scatter.png"
plt.savefig(out, dpi=300)
plt.close()

print("Saved:", out)
