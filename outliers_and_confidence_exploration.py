from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN_DIR = Path("test_run_naturalistic")
OUT_DIR = Path("figures_test_run_naturalistic")
OUT_DIR.mkdir(exist_ok=True)

MAX_EXCERPTS = 6
NO_COMMIT_VALUE = MAX_EXCERPTS + 1

rows = []
turn_rows = []

for path in sorted(RUN_DIR.glob("*.json")):
    with path.open("r", encoding="utf-8") as f:
        run = json.load(f)

    cond = run.get("condition", {})
    agent = cond.get("agent_type")
    reassurance = cond.get("reassurance_type")
    sequence = run.get("sequence_name", Path(cond.get("sequence_file", "unknown")).stem)
    condition = f"{agent} / {reassurance}"

    turns = run.get("turns", [])
    confs = [t.get("parsed_confidence") for t in turns if t.get("parsed_confidence") is not None]

    toc = run.get("toc_turn")
    toc_plot = toc if toc is not None else NO_COMMIT_VALUE

    rows.append({
        "file": path.name,
        "agent": agent,
        "reassurance": reassurance,
        "sequence": sequence,
        "condition": condition,
        "toc": toc_plot,
        "evidence_requests": run.get("evidence_requests"),
        "accuracy": run.get("accuracy"),
        "final_confidence": run.get("final_confidence"),
        "confidence_turn_2": confs[1] if len(confs) >= 2 else np.nan,
        "mean_abs_confidence_change": np.mean(np.abs(np.diff(confs))) if len(confs) >= 2 else np.nan,
        "max_abs_confidence_jump": np.max(np.abs(np.diff(confs))) if len(confs) >= 2 else np.nan,
        "sd_confidence": np.std(confs, ddof=1) if len(confs) >= 2 else np.nan,
        "confidence_range": max(confs) - min(confs) if len(confs) >= 2 else np.nan,
    })

    for t in turns:
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
    raise SystemExit("No JSON files found in test_run_naturalistic/")

condition_order = [
    "nonjtc / miscalibrated",
    "jtc / miscalibrated",
    "nonjtc / calibrated",
    "jtc / calibrated",
]

def sem(x):
    x = pd.Series(x).dropna()
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0

# =====================================================
# 1. Correlation heatmap
# =====================================================

heat_cols = [
    "mean_abs_confidence_change",
    "max_abs_confidence_jump",
    "sd_confidence",
    "confidence_turn_2",
    "confidence_range",
    "toc",
    "evidence_requests",
    "accuracy",
]

corr = df[heat_cols].corr()

plt.figure(figsize=(8.5, 7))
im = plt.imshow(corr, vmin=-1, vmax=1, cmap="viridis")
plt.colorbar(im, label="Pearson r")

plt.xticks(range(len(heat_cols)), heat_cols, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(heat_cols)), heat_cols, fontsize=8)

for i in range(len(heat_cols)):
    for j in range(len(heat_cols)):
        val = corr.iloc[i, j]
        if pd.notna(val):
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

plt.title("Correlation matrix: confidence volatility, TOC, ES")
plt.tight_layout()
plt.savefig(OUT_DIR / "correlation_matrix_confidence_volatility_toc_es.png", dpi=300)
plt.close()

# =====================================================
# 2. Confidence volatility bar plot
# =====================================================

vol_summary = (
    df.groupby("condition")["mean_abs_confidence_change"]
    .agg(["mean", sem, "count"])
    .reindex(condition_order)
)

plt.figure(figsize=(8, 5))
x = np.arange(len(vol_summary))

plt.bar(x, vol_summary["mean"], yerr=vol_summary["sem"], capsize=5)

for i, row in vol_summary.iterrows():
    if pd.notna(row["mean"]):
        plt.text(
            list(vol_summary.index).index(i),
            row["mean"],
            f"n={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.xticks(x, vol_summary.index, rotation=15, ha="right")
plt.ylabel("Mean absolute confidence change")
plt.title("GPT-5.4 runs: confidence volatility")
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_volatility_by_condition.png", dpi=300)
plt.close()

# =====================================================
# 3. Early certainty, TOC, accuracy stacked figure
# =====================================================

metrics = [
    ("confidence_turn_2", "Confidence at turn 2", "Confidence at turn 2"),
    ("toc", "Turn of commitment", "Turn of commitment"),
    ("accuracy", "Accuracy", "Accuracy"),
]

fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

for ax, (metric, ylabel, title) in zip(axes, metrics):
    s = (
        df.groupby("condition")[metric]
        .agg(["mean", sem, "count"])
        .reindex(condition_order)
    )

    x = np.arange(len(s))

    ax.errorbar(
        x,
        s["mean"],
        yerr=s["sem"],
        marker="o",
        linewidth=2,
        capsize=4,
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

axes[-1].set_xticks(np.arange(len(condition_order)))
axes[-1].set_xticklabels(condition_order, rotation=15, ha="right")

fig.suptitle("GPT-5.4: Early certainty, commitment timing, and accuracy", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "early_certainty_commitment_accuracy.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# 4. Individual confidence trajectories by condition
# =====================================================

fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True, sharey=True)

for ax, condition in zip(axes, condition_order):
    sub = turn_df[turn_df["condition"] == condition]

    for file, file_df in sub.groupby("file"):
        file_df = file_df.sort_values("turn")
        ax.plot(
            file_df["turn"],
            file_df["confidence"],
            linewidth=1,
            alpha=0.15,
        )

    s = (
        sub.groupby("turn")["confidence"]
        .agg(["mean", sem])
        .reset_index()
    )

    ax.errorbar(
        s["turn"],
        s["mean"],
        yerr=s["sem"],
        marker="o",
        linewidth=2.5,
        capsize=4,
    )

    ax.set_title(condition)
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

axes[-1].set_xlabel("Turn index")
axes[-1].set_xticks(range(1, MAX_EXCERPTS + 1))

fig.suptitle("GPT-5.4 runs: individual confidence trajectories", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "individual_confidence_trajectories.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# Save summaries
# =====================================================

df.to_csv(OUT_DIR / "run_level_outliers_confidence_metrics.csv", index=False)
turn_df.to_csv(OUT_DIR / "turn_level_confidence_trajectories.csv", index=False)

summary = (
    df.groupby(["agent", "reassurance"])
    .agg(
        toc_mean=("toc", "mean"),
        toc_sem=("toc", sem),
        es_mean=("evidence_requests", "mean"),
        es_sem=("evidence_requests", sem),
        accuracy_mean=("accuracy", "mean"),
        confidence_turn_2_mean=("confidence_turn_2", "mean"),
        confidence_volatility_mean=("mean_abs_confidence_change", "mean"),
        confidence_volatility_sem=("mean_abs_confidence_change", sem),
        n=("file", "count"),
    )
    .reset_index()
)

summary.to_csv(OUT_DIR / "summary_outliers_confidence_metrics.csv", index=False)

print("Saved plots to:", OUT_DIR)
print(summary.round(3))