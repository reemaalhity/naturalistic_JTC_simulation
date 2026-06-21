from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs_latest")   # change to runs_final if needed
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

rows = []

for path in RUNS_DIR.glob("*.json"):
    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)

    model = r.get("model_name", "")
    if model != "openai/gpt-5.4":
        continue

    cond = r.get("condition", {})

    rows.append({
        "model": model,
        "agent_type": cond.get("agent_type"),
        "reassurance_type": cond.get("reassurance_type"),
        "sequence_file": cond.get("sequence_file"),
        "rep": cond.get("rep"),
        "toc_turn": r.get("toc_turn"),
        "accuracy": r.get("accuracy"),
        "final_guess": r.get("final_guess"),
    })

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No GPT-5.4 runs found. Check model name or RUNS_DIR.")

# Keep only runs with valid TOC and accuracy
df_clean = df.dropna(subset=["toc_turn", "accuracy"]).copy()
df_clean["toc_turn"] = df_clean["toc_turn"].astype(int)
df_clean["accuracy"] = df_clean["accuracy"].astype(int)

print("\nCounts by TOC and accuracy:")
print(pd.crosstab(df_clean["toc_turn"], df_clean["accuracy"]))

print("\nMean accuracy by TOC:")
summary = df_clean.groupby("toc_turn")["accuracy"].agg(["count", "mean", "std"]).reset_index()
print(summary)

print("\nMean accuracy by condition and TOC:")
condition_summary = (
    df_clean
    .groupby(["agent_type", "reassurance_type", "toc_turn"])["accuracy"]
    .agg(["count", "mean", "std"])
    .reset_index()
)
print(condition_summary)

# Save summaries
summary.to_csv(OUT_DIR / "gpt54_accuracy_by_toc_summary.csv", index=False)
condition_summary.to_csv(OUT_DIR / "gpt54_accuracy_by_condition_toc_summary.csv", index=False)

# Plot 1: simple accuracy by TOC
plt.figure(figsize=(7, 5))
means = df_clean.groupby("toc_turn")["accuracy"].mean()
counts = df_clean.groupby("toc_turn")["accuracy"].count()

plt.bar(means.index.astype(str), means.values)
plt.ylim(0, 1.05)
plt.xlabel("Turn of commitment (TOC)")
plt.ylabel("Mean final author-classification accuracy")
plt.title("GPT-5.4: Accuracy by Turn of Commitment")

for i, (toc, mean_val) in enumerate(means.items()):
    n = counts.loc[toc]
    plt.text(i, mean_val + 0.03, f"n={n}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_accuracy_by_toc.png", dpi=300)
plt.close()

# Plot 2: condition-level accuracy by TOC
df_clean["condition"] = df_clean["agent_type"] + " / " + df_clean["reassurance_type"]

pivot = (
    df_clean
    .groupby(["condition", "toc_turn"])["accuracy"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 5))

for condition, sub in pivot.groupby("condition"):
    sub = sub.sort_values("toc_turn")
    plt.plot(sub["toc_turn"], sub["accuracy"], marker="o", label=condition)

plt.ylim(0, 1.05)
plt.xlabel("Turn of commitment (TOC)")
plt.ylabel("Mean final author-classification accuracy")
plt.title("GPT-5.4: Accuracy by TOC across conditions")
plt.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_accuracy_by_toc_conditions.png", dpi=300)
plt.close()

print("\nSaved:")
print(OUT_DIR / "gpt54_accuracy_by_toc.png")
print(OUT_DIR / "gpt54_accuracy_by_toc_conditions.png")
print(OUT_DIR / "gpt54_accuracy_by_toc_summary.csv")
print(OUT_DIR / "gpt54_accuracy_by_condition_toc_summary.csv")
