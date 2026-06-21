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
    turns = r.get("turns", [])

    commit_turn = None
    commit_conf = None

    for t in turns:
        action = str(t.get("parsed_action", "")).lower()
        if "commit" in action:
            commit_turn = t.get("turn_index")
            commit_conf = t.get("parsed_confidence")
            break

    # fallback if top-level toc_turn exists but commit action not found
    if commit_turn is None:
        commit_turn = r.get("toc_turn")
        for t in turns:
            if t.get("turn_index") == commit_turn:
                commit_conf = t.get("parsed_confidence")
                break

    if commit_turn is None or commit_conf is None:
        continue

    rows.append({
        "agent_type": cond.get("agent_type"),
        "reassurance_type": cond.get("reassurance_type"),
        "sequence_file": cond.get("sequence_file"),
        "toc_turn": commit_turn,
        "commit_confidence": commit_conf,
        "accuracy": r.get("accuracy"),
    })

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No commit confidence values found. Check runs folder or JSON structure.")

df = df.dropna(subset=["toc_turn", "commit_confidence", "accuracy"]).copy()

df["toc_turn"] = df["toc_turn"].astype(int)
df["commit_confidence"] = df["commit_confidence"].astype(float)
df["accuracy"] = df["accuracy"].astype(int)

df["sequence"] = df["sequence_file"].str.extract(r'(sequence[_\-]?\d)', expand=False)
df["sequence"] = df["sequence"].fillna(df["sequence_file"])

df["condition"] = df["agent_type"] + " / " + df["reassurance_type"]

# Save raw extracted table
df.to_csv(OUT_DIR / "gpt54_commit_confidence_accuracy_raw.csv", index=False)

# Summary: confidence by accuracy
summary_acc = (
    df.groupby(["accuracy"])["commit_confidence"]
    .agg(["mean", "count", "std"])
    .reset_index()
)
summary_acc.to_csv(OUT_DIR / "gpt54_commit_confidence_by_accuracy.csv", index=False)

print("\nCommit confidence by accuracy:")
print(summary_acc)

# Summary: confidence by condition, sequence, accuracy
summary = (
    df.groupby(["condition", "sequence", "accuracy"])["commit_confidence"]
    .agg(["mean", "count", "std"])
    .reset_index()
)
summary.to_csv(OUT_DIR / "gpt54_commit_confidence_by_condition_sequence_accuracy.csv", index=False)

print("\nCommit confidence by condition / sequence / accuracy:")
print(summary)

# Plot 1: confidence by accuracy
plt.figure(figsize=(6, 5))
means = df.groupby("accuracy")["commit_confidence"].mean()
counts = df.groupby("accuracy")["commit_confidence"].count()

labels = ["Incorrect", "Correct"]
plt.bar(labels, [means.get(0, 0), means.get(1, 0)])

for i, acc in enumerate([0, 1]):
    if acc in counts.index:
        plt.text(i, means.loc[acc] + 1, f"n={counts.loc[acc]}", ha="center", fontsize=9)

plt.ylim(0, 100)
plt.ylabel("Mean confidence at commitment")
plt.title("GPT-5.4: Commit confidence by accuracy")
plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_commit_confidence_by_accuracy.png", dpi=300)
plt.close()

# Plot 2: confidence by TOC, split by accuracy
plt.figure(figsize=(7, 5))

for acc_val, sub in df.groupby("accuracy"):
    grouped = sub.groupby("toc_turn")["commit_confidence"].mean().reset_index()
    label = "Correct" if acc_val == 1 else "Incorrect"
    plt.plot(grouped["toc_turn"], grouped["commit_confidence"], marker="o", linewidth=2, label=label)

plt.ylim(0, 100)
plt.xlabel("Turn of commitment (TOC)")
plt.ylabel("Mean confidence at commitment")
plt.title("GPT-5.4: Commit confidence by TOC and accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_commit_confidence_by_toc_accuracy.png", dpi=300)
plt.close()

# Plot 3: condition panels, confidence by TOC split by sequence
plot_summary = (
    df.groupby(["condition", "sequence", "toc_turn"])["commit_confidence"]
    .agg(["mean", "count"])
    .reset_index()
)

conditions = plot_summary["condition"].unique()

fig, axes = plt.subplots(
    len(conditions),
    1,
    figsize=(8, 4 * len(conditions)),
    sharex=True
)

if len(conditions) == 1:
    axes = [axes]

for ax, condition in zip(axes, conditions):
    sub = plot_summary[plot_summary["condition"] == condition]

    for sequence, seq_sub in sub.groupby("sequence"):
        seq_sub = seq_sub.sort_values("toc_turn")
        ax.plot(
            seq_sub["toc_turn"],
            seq_sub["mean"],
            marker="o",
            linewidth=2,
            label=sequence
        )

        for _, row in seq_sub.iterrows():
            ax.text(
                row["toc_turn"],
                row["mean"] + 1,
                f"n={int(row['count'])}",
                ha="center",
                fontsize=8
            )

    ax.set_title(condition)
    ax.set_ylabel("Mean confidence")
    ax.set_ylim(0, 100)
    ax.legend(title="Sequence")

axes[-1].set_xlabel("Turn of commitment (TOC)")

plt.suptitle(
    "GPT-5.4: Confidence at commitment by TOC split by sequence",
    y=0.995,
    fontsize=14
)

plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_commit_confidence_by_sequence_toc.png", dpi=300)
plt.close()

print("\nSaved:")
print(OUT_DIR / "gpt54_commit_confidence_by_accuracy.png")
print(OUT_DIR / "gpt54_commit_confidence_by_toc_accuracy.png")
print(OUT_DIR / "gpt54_commit_confidence_by_sequence_toc.png")
