from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs_latest")   # change if needed
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

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No GPT-5.4 runs found.")

df = df.dropna(subset=["toc_turn", "accuracy"]).copy()

df["toc_turn"] = df["toc_turn"].astype(int)
df["accuracy"] = df["accuracy"].astype(int)

# Simplify sequence labels
df["sequence"] = df["sequence_file"].str.extract(r'(sequence[_\-]?\d)', expand=False)
df["sequence"] = df["sequence"].fillna(df["sequence_file"])

# Create condition label
df["condition"] = (
    df["agent_type"] + " / " + df["reassurance_type"]
)

# Aggregate
summary = (
    df.groupby(["condition", "sequence", "toc_turn"])["accuracy"]
    .agg(["mean", "count"])
    .reset_index()
)

print("\nAccuracy by sequence and TOC:\n")
print(summary)

summary.to_csv(
    OUT_DIR / "gpt54_accuracy_by_sequence_toc.csv",
    index=False
)

# Plot
conditions = summary["condition"].unique()

fig, axes = plt.subplots(
    len(conditions),
    1,
    figsize=(8, 4 * len(conditions)),
    sharex=True
)

if len(conditions) == 1:
    axes = [axes]

for ax, condition in zip(axes, conditions):

    sub = summary[summary["condition"] == condition]

    for sequence, seq_sub in sub.groupby("sequence"):

        seq_sub = seq_sub.sort_values("toc_turn")

        ax.plot(
            seq_sub["toc_turn"],
            seq_sub["mean"],
            marker="o",
            linewidth=2,
            label=sequence
        )

        # annotate n
        for _, row in seq_sub.iterrows():
            ax.text(
                row["toc_turn"],
                row["mean"] + 0.03,
                f"n={int(row['count'])}",
                ha="center",
                fontsize=8
            )

    ax.set_title(f"{condition}")
    ax.set_ylabel("Mean accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Sequence")

axes[-1].set_xlabel("Turn of commitment (TOC)")

plt.suptitle(
    "GPT-5.4: Accuracy by TOC split by sequence",
    y=0.995,
    fontsize=14
)

plt.tight_layout()

outpath = OUT_DIR / "gpt54_accuracy_by_sequence_toc.png"

plt.savefig(outpath, dpi=300)
plt.close()

print(f"\nSaved figure:\n{outpath}")
