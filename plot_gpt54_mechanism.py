from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs_latest")  # change if needed
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

rows = []

for path in RUNS_DIR.glob("*.json"):
    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)

    if r.get("model_name", "") != "openai/gpt-5.4":
        continue

    cond = r.get("condition", {})

    rows.append({
        "agent_type": cond.get("agent_type"),
        "reassurance_type": cond.get("reassurance_type"),
        "sequence_file": cond.get("sequence_file"),
        "toc_turn": r.get("toc_turn"),
        "accuracy": r.get("accuracy"),
    })

df = pd.DataFrame(rows).dropna(subset=["toc_turn", "accuracy"])
df["toc_turn"] = df["toc_turn"].astype(int)
df["accuracy"] = df["accuracy"].astype(int)
df["early_commit"] = df["toc_turn"].eq(2)
df["commit_group"] = df["toc_turn"].apply(lambda x: "TOC = 2" if x == 2 else "TOC > 2")

# ---------- Figure A: early commitment rate ----------
early = (
    df.groupby(["agent_type", "reassurance_type"])["early_commit"]
    .agg(["mean", "count"])
    .reset_index()
)

early.to_csv(OUT_DIR / "gpt54_early_commitment_by_condition.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 5))

labels = []
values = []
counts = []

for _, row in early.iterrows():
    labels.append(f"{row['agent_type']}\n{row['reassurance_type']}")
    values.append(row["mean"])
    counts.append(row["count"])

ax.bar(labels, values)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Proportion of runs with TOC = 2")
ax.set_title("GPT-5.4: Early commitment by condition")

for i, (v, n) in enumerate(zip(values, counts)):
    ax.text(i, v + 0.03, f"n={n}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_early_commitment_by_condition.png", dpi=300)
plt.close()

# ---------- Figure B: accuracy by early vs later commitment ----------
acc = (
    df.groupby(["reassurance_type", "commit_group"])["accuracy"]
    .agg(["mean", "count"])
    .reset_index()
)

acc.to_csv(OUT_DIR / "gpt54_accuracy_by_commit_group.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 5))

pivot = acc.pivot(index="commit_group", columns="reassurance_type", values="mean")
pivot.plot(kind="bar", ax=ax)

ax.set_ylim(0, 1.05)
ax.set_ylabel("Mean accuracy")
ax.set_xlabel("Commitment timing")
ax.set_title("GPT-5.4: Accuracy by early vs later commitment")
ax.legend(title="Reassurance type")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_accuracy_by_commit_group.png", dpi=300)
plt.close()

# ---------- Print summaries ----------
print("\nEarly commitment rate by condition:")
print(early)

print("\nAccuracy by reassurance and commitment group:")
print(acc)

print("\nSaved figures to:")
print(OUT_DIR / "gpt54_early_commitment_by_condition.png")
print(OUT_DIR / "gpt54_accuracy_by_commit_group.png")
