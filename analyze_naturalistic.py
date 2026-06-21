from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN_DIR = Path("test_run_naturalistic")
OUT_DIR = Path("figures_naturalistic")
OUT_DIR.mkdir(exist_ok=True)

MAX_EXCERPTS = 6
NO_COMMIT_VALUE = MAX_EXCERPTS + 1

rows = []
turn_rows = []

for path in sorted(RUN_DIR.glob("*.json")):
    with open(path, "r", encoding="utf-8") as f:
        run = json.load(f)

    cond = run.get("condition", {})
    agent = cond.get("agent_type")
    reassurance = cond.get("reassurance_type")
    sequence = run.get("sequence_name")

    condition = f"{agent}_{reassurance}"

    toc = run.get("toc_turn")
    toc_plot = toc if toc is not None else NO_COMMIT_VALUE

    rows.append({
        "file": path.name,
        "agent": agent,
        "reassurance": reassurance,
        "sequence": sequence,
        "condition": condition,
        "toc": toc_plot,
        "es": run.get("evidence_requests"),
        "accuracy": run.get("accuracy"),
        "final_confidence": run.get("final_confidence"),
    })

    for t in run.get("turns", []):
        conf = t.get("parsed_confidence")
        if conf is not None:
            turn_rows.append({
                "file": path.name,
                "agent": agent,
                "reassurance": reassurance,
                "sequence": sequence,
                "condition": condition,
                "turn": t.get("turn_index"),
                "confidence": conf,
            })

df = pd.DataFrame(rows)
turn_df = pd.DataFrame(turn_rows)

if df.empty:
    raise SystemExit("No runs found in test_run_naturalistic")

condition_order = [
    "jtc_calibrated",
    "jtc_miscalibrated",
    "nonjtc_calibrated",
    "nonjtc_miscalibrated",
]

condition_labels = [
    "JTC\nCal",
    "JTC\nMiscal",
    "NonJTC\nCal",
    "NonJTC\nMiscal",
]

df["condition"] = pd.Categorical(
    df["condition"],
    categories=condition_order,
    ordered=True,
)

turn_df["condition"] = pd.Categorical(
    turn_df["condition"],
    categories=condition_order,
    ordered=True,
)

def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return 0
    return x.std(ddof=1) / np.sqrt(len(x))

# =====================================================
# 1. CONFIDENCE TRAJECTORY
# =====================================================

plt.figure(figsize=(9, 5.5))

for condition in condition_order:
    sub = turn_df[turn_df["condition"] == condition]

    if sub.empty:
        continue

    summary = (
        sub.groupby("turn")["confidence"]
        .agg(["mean", sem, "count"])
        .reset_index()
    )

    plt.errorbar(
        summary["turn"],
        summary["mean"],
        yerr=summary["sem"],
        marker="o",
        capsize=4,
        linewidth=2,
        label=condition.replace("_", " "),
    )

plt.axvline(1, linestyle="--", alpha=0.35)
plt.xlabel("Turn")
plt.ylabel("Mean confidence")
plt.title("Confidence trajectories by condition")
plt.xticks(range(1, MAX_EXCERPTS + 1))
plt.ylim(40, 100)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_trajectory_by_condition.png", dpi=300)
plt.close()

# =====================================================
# Helper for bar plots
# =====================================================

def make_bar(metric, ylabel, title, filename, ylim=None):
    summary = (
        df.groupby("condition", observed=False)[metric]
        .agg(["mean", sem, "count"])
        .reset_index()
    )

    summary = summary.dropna(subset=["condition"])

    x = np.arange(len(summary))

    plt.figure(figsize=(7, 5))

    plt.bar(
        x,
        summary["mean"],
        yerr=summary["sem"],
        capsize=5,
    )

    for i, row in summary.iterrows():
        plt.text(
            i,
            row["mean"] + 0.03 * (summary["mean"].max() if summary["mean"].max() else 1),
            f"n={int(row['count'])}",
            ha="center",
            fontsize=8,
        )

    plt.xticks(x, condition_labels[:len(summary)])
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()

# =====================================================
# 2. TOC
# =====================================================

make_bar(
    metric="toc",
    ylabel="Mean turn of commitment",
    title="Turn of commitment by condition",
    filename="toc_by_condition.png",
    ylim=(0, NO_COMMIT_VALUE + 0.5),
)

# =====================================================
# 3. ES
# =====================================================

make_bar(
    metric="es",
    ylabel="Mean evidence requests",
    title="Evidence sampling by condition",
    filename="evidence_sampling_by_condition.png",
    ylim=(0, MAX_EXCERPTS),
)

# =====================================================
# 4. Accuracy
# =====================================================

make_bar(
    metric="accuracy",
    ylabel="Mean accuracy",
    title="Accuracy by condition",
    filename="accuracy_by_condition.png",
    ylim=(0, 1.05),
)

# =====================================================
# 5. Final confidence
# =====================================================

make_bar(
    metric="final_confidence",
    ylabel="Mean final confidence",
    title="Final confidence by condition",
    filename="final_confidence_by_condition.png",
    ylim=(40, 100),
)

# =====================================================
# 6. TOC distribution
# =====================================================

plt.figure(figsize=(8, 5.5))

for condition in condition_order:
    sub = df[df["condition"] == condition]
    counts = sub["toc"].value_counts().sort_index()
    plt.plot(
        counts.index,
        counts.values,
        marker="o",
        linewidth=2,
        label=condition.replace("_", " "),
    )

plt.xlabel("Turn of commitment")
plt.ylabel("Count")
plt.title("Distribution of commitment turns")
plt.xticks(range(1, NO_COMMIT_VALUE + 1))
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "toc_distribution_by_condition.png", dpi=300)
plt.close()

# =====================================================
# Save summary CSVs
# =====================================================

summary = (
    df.groupby(["agent", "reassurance"], observed=False)
    .agg(
        toc_mean=("toc", "mean"),
        toc_sem=("toc", sem),
        es_mean=("es", "mean"),
        es_sem=("es", sem),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sem=("accuracy", sem),
        final_confidence_mean=("final_confidence", "mean"),
        final_confidence_sem=("final_confidence", sem),
        n=("file", "count"),
    )
    .reset_index()
)

summary.to_csv(OUT_DIR / "naturalistic_summary.csv", index=False)

turn_summary = (
    turn_df.groupby(["agent", "reassurance", "turn"], observed=False)
    .agg(
        confidence_mean=("confidence", "mean"),
        confidence_sem=("confidence", sem),
        n=("file", "count"),
    )
    .reset_index()
)

turn_summary.to_csv(OUT_DIR / "confidence_trajectory_summary.csv", index=False)

print("\nSaved plots to:", OUT_DIR)
print(summary.round(3))