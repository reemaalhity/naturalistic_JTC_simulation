from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

RUN_DIR = Path("runs_naturalistic_latest")
OUT_DIR = Path("figures_naturalistic_full")
OUT_DIR.mkdir(exist_ok=True)

MAX_EXCERPTS = 6
NO_COMMIT_VALUE = MAX_EXCERPTS + 1

# =====================================================
# LOAD RUNS
# =====================================================

rows = []
turn_rows = []

for path in sorted(RUN_DIR.glob("*.json")):
    with path.open("r", encoding="utf-8") as f:
        run = json.load(f)

    cond = run.get("condition", {})
    turns = run.get("turns", [])

    agent = cond.get("agent_type")
    reassurance = cond.get("reassurance_type")
    sequence = run.get("sequence_name")
    model = run.get("model_name")

    condition = f"{agent}_{reassurance}"

    toc = run.get("toc_turn")
    toc_plot = toc if toc is not None else NO_COMMIT_VALUE

    confs = [
        t.get("parsed_confidence")
        for t in turns
        if t.get("parsed_confidence") is not None
    ]

    first_conf = confs[0] if len(confs) >= 1 else np.nan
    second_conf = confs[1] if len(confs) >= 2 else np.nan
    final_conf = run.get("final_confidence")

    conf_delta_1_2 = (
        second_conf - first_conf
        if len(confs) >= 2
        else np.nan
    )

    conf_range = (
        max(confs) - min(confs)
        if len(confs) >= 2
        else np.nan
    )

    mean_abs_conf_change = (
        np.mean(np.abs(np.diff(confs)))
        if len(confs) >= 2
        else np.nan
    )

    max_abs_conf_jump = (
        np.max(np.abs(np.diff(confs)))
        if len(confs) >= 2
        else np.nan
    )

    rows.append({
        "file": path.name,
        "model": model,
        "agent": agent,
        "reassurance": reassurance,
        "sequence": sequence,
        "condition": condition,
        "toc": toc_plot,
        "evidence_requests": run.get("evidence_requests"),
        "accuracy": run.get("accuracy"),
        "first_confidence": first_conf,
        "second_confidence": second_conf,
        "final_confidence": final_conf,
        "confidence_delta_1_to_2": conf_delta_1_2,
        "confidence_range": conf_range,
        "mean_abs_confidence_change": mean_abs_conf_change,
        "max_abs_confidence_jump": max_abs_conf_jump,
        "early_commit_2": int(toc_plot <= 2),
        "early_commit_3": int(toc_plot <= 3),
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
    raise SystemExit("No JSON files found in runs_naturalistic_latest")

condition_order = [
    "jtc_calibrated",
    "jtc_miscalibrated",
    "nonjtc_calibrated",
    "nonjtc_miscalibrated",
]

condition_labels = {
    "jtc_calibrated": "JTC calibrated",
    "jtc_miscalibrated": "JTC miscalibrated",
    "nonjtc_calibrated": "Non-JTC calibrated",
    "nonjtc_miscalibrated": "Non-JTC miscalibrated",
}

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
# SAVE RAW TABLES
# =====================================================

df.to_csv(OUT_DIR / "run_level_metrics.csv", index=False)
turn_df.to_csv(OUT_DIR / "turn_level_confidence.csv", index=False)

# =====================================================
# PLOT 1: CONFIDENCE TRAJECTORY COLLAPSED ACROSS SEQUENCES
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
        label=condition_labels[condition],
    )

plt.xlabel("Turn")
plt.ylabel("Mean confidence")
plt.title("Confidence trajectories collapsed across sequences")
plt.xticks(range(1, MAX_EXCERPTS + 1))
plt.ylim(40, 100)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_trajectory_collapsed.png", dpi=300)
plt.close()

# =====================================================
# PLOT 2: CONFIDENCE TRAJECTORY BY SEQUENCE
# =====================================================

sequences = sorted(turn_df["sequence"].dropna().unique())

fig, axes = plt.subplots(
    1,
    len(sequences),
    figsize=(6 * len(sequences), 5),
    sharey=True,
)

if len(sequences) == 1:
    axes = [axes]

for ax, seq in zip(axes, sequences):
    seq_sub = turn_df[turn_df["sequence"] == seq]

    for condition in condition_order:
        sub = seq_sub[seq_sub["condition"] == condition]
        if sub.empty:
            continue

        summary = (
            sub.groupby("turn")["confidence"]
            .agg(["mean", sem, "count"])
            .reset_index()
        )

        ax.errorbar(
            summary["turn"],
            summary["mean"],
            yerr=summary["sem"],
            marker="o",
            capsize=4,
            linewidth=2,
            label=condition_labels[condition],
        )

    ax.set_title(seq)
    ax.set_xlabel("Turn")
    ax.set_xticks(range(1, MAX_EXCERPTS + 1))
    ax.set_ylim(40, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

axes[0].set_ylabel("Mean confidence")
axes[-1].legend(frameon=False, fontsize=8)

plt.suptitle("Confidence trajectories by sequence", y=1.03)
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_trajectory_by_sequence.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# PLOT 3: CONDITION × SEQUENCE TRAJECTORY GRID
# =====================================================

fig, axes = plt.subplots(
    len(condition_order),
    len(sequences),
    figsize=(5.5 * len(sequences), 3.2 * len(condition_order)),
    sharex=True,
    sharey=True,
)

if len(sequences) == 1:
    axes = np.array([[ax] for ax in axes])

for row_idx, condition in enumerate(condition_order):
    for col_idx, seq in enumerate(sequences):
        ax = axes[row_idx, col_idx]

        sub = turn_df[
            (turn_df["condition"] == condition)
            & (turn_df["sequence"] == seq)
        ]

        if not sub.empty:
            summary = (
                sub.groupby("turn")["confidence"]
                .agg(["mean", sem, "count"])
                .reset_index()
            )

            ax.errorbar(
                summary["turn"],
                summary["mean"],
                yerr=summary["sem"],
                marker="o",
                capsize=4,
                linewidth=2,
            )

        ax.set_title(f"{condition_labels[condition]}\n{seq}", fontsize=10)
        ax.set_xticks(range(1, MAX_EXCERPTS + 1))
        ax.set_ylim(40, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if col_idx == 0:
            ax.set_ylabel("Confidence")
        if row_idx == len(condition_order) - 1:
            ax.set_xlabel("Turn")

plt.suptitle("Confidence trajectories separated by condition and sequence", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_trajectory_condition_sequence_grid.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# SUMMARY TABLE
# =====================================================

summary = (
    df.groupby(["agent", "reassurance", "sequence"], observed=False)
    .agg(
        toc_mean=("toc", "mean"),
        toc_sem=("toc", sem),
        es_mean=("evidence_requests", "mean"),
        es_sem=("evidence_requests", sem),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sem=("accuracy", sem),
        final_confidence_mean=("final_confidence", "mean"),
        final_confidence_sem=("final_confidence", sem),
        confidence_range_mean=("confidence_range", "mean"),
        confidence_delta_1_to_2_mean=("confidence_delta_1_to_2", "mean"),
        mean_abs_confidence_change_mean=("mean_abs_confidence_change", "mean"),
        early_commit_2_rate=("early_commit_2", "mean"),
        early_commit_3_rate=("early_commit_3", "mean"),
        n=("file", "count"),
    )
    .reset_index()
)

summary.to_csv(OUT_DIR / "summary_by_agent_reassurance_sequence.csv", index=False)

summary_collapsed = (
    df.groupby(["agent", "reassurance"], observed=False)
    .agg(
        toc_mean=("toc", "mean"),
        toc_sem=("toc", sem),
        es_mean=("evidence_requests", "mean"),
        es_sem=("evidence_requests", sem),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sem=("accuracy", sem),
        final_confidence_mean=("final_confidence", "mean"),
        final_confidence_sem=("final_confidence", sem),
        confidence_range_mean=("confidence_range", "mean"),
        confidence_delta_1_to_2_mean=("confidence_delta_1_to_2", "mean"),
        mean_abs_confidence_change_mean=("mean_abs_confidence_change", "mean"),
        early_commit_2_rate=("early_commit_2", "mean"),
        early_commit_3_rate=("early_commit_3", "mean"),
        n=("file", "count"),
    )
    .reset_index()
)

summary_collapsed.to_csv(OUT_DIR / "summary_collapsed_across_sequences.csv", index=False)

# =====================================================
# HEATMAP 1: CORRELATION MATRIX TESTING EVERYTHING
# =====================================================

heat_df = df.copy()

heat_df["agent_jtc"] = (heat_df["agent"] == "jtc").astype(int)
heat_df["agent_nonjtc"] = (heat_df["agent"] == "nonjtc").astype(int)
heat_df["reassurance_calibrated"] = (heat_df["reassurance"] == "calibrated").astype(int)
heat_df["reassurance_miscalibrated"] = (heat_df["reassurance"] == "miscalibrated").astype(int)

for seq in sorted(heat_df["sequence"].dropna().unique()):
    heat_df[f"sequence_{seq}"] = (heat_df["sequence"] == seq).astype(int)

corr_cols = [
    "agent_jtc",
    "agent_nonjtc",
    "reassurance_calibrated",
    "reassurance_miscalibrated",
    "toc",
    "evidence_requests",
    "accuracy",
    "final_confidence",
    "first_confidence",
    "second_confidence",
    "confidence_delta_1_to_2",
    "confidence_range",
    "mean_abs_confidence_change",
    "max_abs_confidence_jump",
    "early_commit_2",
    "early_commit_3",
]

corr_cols += [c for c in heat_df.columns if c.startswith("sequence_")]

corr_data = heat_df[corr_cols].apply(pd.to_numeric, errors="coerce")

corr = corr_data.corr()
pvals = pd.DataFrame(np.nan, index=corr.columns, columns=corr.columns)

if SCIPY_AVAILABLE:
    for c1 in corr.columns:
        for c2 in corr.columns:
            sub = corr_data[[c1, c2]].dropna()
            if len(sub) >= 3 and sub[c1].nunique() > 1 and sub[c2].nunique() > 1:
                _, p = pearsonr(sub[c1], sub[c2])
                pvals.loc[c1, c2] = p

corr.to_csv(OUT_DIR / "big_correlation_matrix_r.csv")
pvals.to_csv(OUT_DIR / "big_correlation_matrix_p.csv")

plt.figure(figsize=(15, 13))
im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")

plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson r")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        r = corr.iloc[i, j]
        if pd.isna(r):
            label = ""
        else:
            stars = ""
            if SCIPY_AVAILABLE:
                p = pvals.iloc[i, j]
                if pd.notna(p):
                    if p < 0.001:
                        stars = "***"
                    elif p < 0.01:
                        stars = "**"
                    elif p < 0.05:
                        stars = "*"
            label = f"{r:.2f}{stars}"

        plt.text(j, i, label, ha="center", va="center", fontsize=6)

plt.title("Big heatmap matrix: correlations across condition codes and outcomes")
plt.tight_layout()
plt.savefig(OUT_DIR / "big_heatmap_correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# HEATMAP 2: CONDITION × METRIC MEANS
# =====================================================

metric_cols = [
    "toc",
    "evidence_requests",
    "accuracy",
    "final_confidence",
    "first_confidence",
    "second_confidence",
    "confidence_delta_1_to_2",
    "confidence_range",
    "mean_abs_confidence_change",
    "max_abs_confidence_jump",
    "early_commit_2",
    "early_commit_3",
]

condition_metric = (
    df.groupby("condition", observed=False)[metric_cols]
    .mean()
    .reindex(condition_order)
)

condition_metric.to_csv(OUT_DIR / "condition_metric_means.csv")

# z-score columns so different scales can be shown together
z = condition_metric.copy()

for col in z.columns:
    col_sd = z[col].std(ddof=0)
    if col_sd == 0 or pd.isna(col_sd):
        z[col] = 0
    else:
        z[col] = (z[col] - z[col].mean()) / col_sd

plt.figure(figsize=(14, 5.5))
im = plt.imshow(z.values, cmap="coolwarm", aspect="auto")

plt.colorbar(im, fraction=0.046, pad=0.04, label="Column z-score")
plt.xticks(range(len(z.columns)), z.columns, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(z.index)), [condition_labels[str(i)] for i in z.index], fontsize=9)

for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        raw_val = condition_metric.iloc[i, j]
        z_val = z.iloc[i, j]
        plt.text(
            j,
            i,
            f"{raw_val:.2f}\n({z_val:.1f}z)",
            ha="center",
            va="center",
            fontsize=7,
        )

plt.title("Condition × metric heatmap")
plt.tight_layout()
plt.savefig(OUT_DIR / "condition_metric_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# SIMPLE BAR PLOTS FOR CORE OUTCOMES
# =====================================================

def make_bar(metric, ylabel, title, filename, ylim=None):
    s = (
        df.groupby("condition", observed=False)[metric]
        .agg(["mean", sem, "count"])
        .reindex(condition_order)
        .reset_index()
    )

    x = np.arange(len(s))

    plt.figure(figsize=(7.5, 5))
    plt.bar(x, s["mean"], yerr=s["sem"], capsize=5)

    for i, row in s.iterrows():
        if pd.notna(row["mean"]):
            plt.text(
                i,
                row["mean"] + 0.03 * max(s["mean"].max(), 1),
                f"n={int(row['count'])}",
                ha="center",
                fontsize=8,
            )

    plt.xticks(x, [condition_labels[c] for c in condition_order], rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()

make_bar("toc", "Mean TOC", "Turn of commitment by condition", "bar_toc_by_condition.png", (0, NO_COMMIT_VALUE + 0.5))
make_bar("evidence_requests", "Mean evidence requests", "Evidence sampling by condition", "bar_es_by_condition.png", (0, MAX_EXCERPTS))
make_bar("accuracy", "Mean accuracy", "Accuracy by condition", "bar_accuracy_by_condition.png", (0, 1.05))
make_bar("final_confidence", "Mean final confidence", "Final confidence by condition", "bar_final_confidence_by_condition.png", (40, 100))
make_bar("early_commit_2", "Proportion committed by turn 2", "Early commitment rate by condition", "bar_early_commit_2_by_condition.png", (0, 1.05))
make_bar("early_commit_3", "Proportion committed by turn 3", "Commitment by turn 3 rate by condition", "bar_early_commit_3_by_condition.png", (0, 1.05))

print("\nSaved figures to:", OUT_DIR)
print("\nKey outputs:")
print(OUT_DIR / "confidence_trajectory_collapsed.png")
print(OUT_DIR / "confidence_trajectory_by_sequence.png")
print(OUT_DIR / "confidence_trajectory_condition_sequence_grid.png")
print(OUT_DIR / "big_heatmap_correlation_matrix.png")
print(OUT_DIR / "condition_metric_heatmap.png")
print(OUT_DIR / "summary_collapsed_across_sequences.csv")
print(OUT_DIR / "summary_by_agent_reassurance_sequence.csv")