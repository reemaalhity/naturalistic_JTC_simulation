from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN_DIR = Path("test_run_naturalistic")
OUT_DIR = Path("figures_test_run_naturalistic_fair")
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

    condition = f"{agent}_{reassurance}"
    label = f"{agent} / {reassurance}"

    turns = run.get("turns", [])
    confs = [
        t.get("parsed_confidence")
        for t in turns
        if t.get("parsed_confidence") is not None
    ]

    toc = run.get("toc_turn")
    toc_plot = toc if toc is not None else NO_COMMIT_VALUE

    diffs = np.diff(confs) if len(confs) >= 2 else []

    rows.append({
        "file": path.name,
        "agent": agent,
        "reassurance": reassurance,
        "sequence": sequence,
        "condition": condition,
        "label": label,
        "toc": toc_plot,
        "es": run.get("evidence_requests"),
        "accuracy": run.get("accuracy"),
        "final_confidence": run.get("final_confidence"),

        # Fairer volatility metrics
        "mean_abs_conf_change_per_transition": (
            np.mean(np.abs(diffs)) if len(diffs) else np.nan
        ),
        "mean_signed_conf_change_per_transition": (
            np.mean(diffs) if len(diffs) else np.nan
        ),

        # Fixed-window early slope metrics
        "conf_turn_1": confs[0] if len(confs) >= 1 else np.nan,
        "conf_turn_2": confs[1] if len(confs) >= 2 else np.nan,
        "conf_turn_3": confs[2] if len(confs) >= 3 else np.nan,
        "delta_conf_1_to_2": confs[1] - confs[0] if len(confs) >= 2 else np.nan,
        "delta_conf_1_to_3": confs[2] - confs[0] if len(confs) >= 3 else np.nan,

        # Descriptive only
        "sd_confidence": np.std(confs, ddof=1) if len(confs) >= 2 else np.nan,
        "confidence_range": max(confs) - min(confs) if len(confs) >= 2 else np.nan,
        "max_abs_conf_jump": np.max(np.abs(diffs)) if len(diffs) else np.nan,
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
                "label": label,
                "turn": t.get("turn_index"),
                "confidence": conf,
            })

df = pd.DataFrame(rows)
turn_df = pd.DataFrame(turn_rows)

if df.empty:
    raise SystemExit("No JSON files found in test_run_naturalistic/")

condition_order = [
    "jtc_calibrated",
    "jtc_miscalibrated",
    "nonjtc_calibrated",
    "nonjtc_miscalibrated",
]

condition_labels = {
    "jtc_calibrated": "JTC Cal",
    "jtc_miscalibrated": "JTC Miscal",
    "nonjtc_calibrated": "NonJTC Cal",
    "nonjtc_miscalibrated": "NonJTC Miscal",
}

def sem(x):
    x = pd.Series(x).dropna()
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0

df.to_csv(OUT_DIR / "run_level_fair_volatility_metrics.csv", index=False)
turn_df.to_csv(OUT_DIR / "turn_level_confidence.csv", index=False)

# =====================================================
# Helper: grouped bar by condition
# =====================================================

def bar_by_condition(metric, ylabel, title, filename, ylim=None):
    s = (
        df.groupby("condition")[metric]
        .agg(["mean", sem, "count"])
        .reindex(condition_order)
    )

    x = np.arange(len(s))

    plt.figure(figsize=(8, 5))
    plt.bar(x, s["mean"], yerr=s["sem"], capsize=5)

    for i, row in enumerate(s.itertuples()):
        if pd.notna(row.mean):
            plt.text(i, row.mean, f"n={int(row.count)}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, [condition_labels[c] for c in condition_order], rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()

bar_by_condition(
    "mean_abs_conf_change_per_transition",
    "Mean absolute confidence change per transition",
    "Fair confidence volatility by condition",
    "fair_volatility_mean_abs_change_by_condition.png",
)

bar_by_condition(
    "delta_conf_1_to_2",
    "Confidence change from turn 1 to 2",
    "Early confidence slope by condition",
    "early_confidence_delta_1_to_2_by_condition.png",
)

bar_by_condition(
    "delta_conf_1_to_3",
    "Confidence change from turn 1 to 3",
    "Early confidence slope from turn 1 to 3",
    "early_confidence_delta_1_to_3_by_condition.png",
)

# =====================================================
# ES and TOC separated by sequence
# =====================================================

def grouped_bar_by_sequence(metric, ylabel, title, filename, ylim=None):
    sequences = sorted(df["sequence"].dropna().unique())
    x = np.arange(len(condition_order))
    width = 0.8 / max(len(sequences), 1)

    plt.figure(figsize=(10, 5.5))

    for i, seq in enumerate(sequences):
        sub = df[df["sequence"] == seq]
        s = (
            sub.groupby("condition")[metric]
            .agg(["mean", sem, "count"])
            .reindex(condition_order)
        )

        offset = (i - (len(sequences) - 1) / 2) * width

        plt.bar(
            x + offset,
            s["mean"],
            width=width,
            yerr=s["sem"],
            capsize=4,
            label=seq,
        )

    plt.xticks(x, [condition_labels[c] for c in condition_order], rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()

grouped_bar_by_sequence(
    "toc",
    "Mean turn of commitment",
    "TOC by condition, separated by sequence",
    "toc_by_condition_separated_by_sequence.png",
    ylim=(0, NO_COMMIT_VALUE + 0.5),
)

grouped_bar_by_sequence(
    "es",
    "Mean evidence requests",
    "Evidence sampling by condition, separated by sequence",
    "es_by_condition_separated_by_sequence.png",
    ylim=(0, MAX_EXCERPTS),
)

# =====================================================
# Confidence trajectories: pooled and by sequence
# =====================================================

plt.figure(figsize=(9, 5.5))

for condition in condition_order:
    sub = turn_df[turn_df["condition"] == condition]
    if sub.empty:
        continue

    s = sub.groupby("turn")["confidence"].agg(["mean", sem]).reset_index()

    plt.errorbar(
        s["turn"],
        s["mean"],
        yerr=s["sem"],
        marker="o",
        capsize=4,
        linewidth=2,
        label=condition_labels[condition],
    )

plt.xlabel("Turn")
plt.ylabel("Mean confidence")
plt.title("Confidence trajectories collapsed across sequences")
plt.xticks(range(1, MAX_EXCERPTS + 1))
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_trajectory_collapsed.png", dpi=300)
plt.close()

sequences = sorted(turn_df["sequence"].dropna().unique())

fig, axes = plt.subplots(1, len(sequences), figsize=(7 * len(sequences), 5.5), sharey=True)
if len(sequences) == 1:
    axes = [axes]

for ax, seq in zip(axes, sequences):
    seq_sub = turn_df[turn_df["sequence"] == seq]

    for condition in condition_order:
        sub = seq_sub[seq_sub["condition"] == condition]
        if sub.empty:
            continue

        s = sub.groupby("turn")["confidence"].agg(["mean", sem]).reset_index()

        ax.errorbar(
            s["turn"],
            s["mean"],
            yerr=s["sem"],
            marker="o",
            capsize=4,
            linewidth=2,
            label=condition_labels[condition],
        )

    ax.set_title(seq)
    ax.set_xlabel("Turn")
    ax.set_xticks(range(1, MAX_EXCERPTS + 1))
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

axes[0].set_ylabel("Mean confidence")
axes[-1].legend(frameon=False, fontsize=8)

plt.suptitle("Confidence trajectories separated by sequence", y=1.03)
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_trajectory_by_sequence.png", dpi=300, bbox_inches="tight")
plt.close()

# =====================================================
# Summary CSVs
# =====================================================

summary = (
    df.groupby(["agent", "reassurance"])
    .agg(
        toc_mean=("toc", "mean"),
        toc_sem=("toc", sem),
        es_mean=("es", "mean"),
        es_sem=("es", sem),
        fair_volatility_mean=("mean_abs_conf_change_per_transition", "mean"),
        fair_volatility_sem=("mean_abs_conf_change_per_transition", sem),
        delta_1_to_2_mean=("delta_conf_1_to_2", "mean"),
        delta_1_to_2_sem=("delta_conf_1_to_2", sem),
        delta_1_to_3_mean=("delta_conf_1_to_3", "mean"),
        delta_1_to_3_sem=("delta_conf_1_to_3", sem),
        n=("file", "count"),
    )
    .reset_index()
)

summary_seq = (
    df.groupby(["agent", "reassurance", "sequence"])
    .agg(
        toc_mean=("toc", "mean"),
        toc_sem=("toc", sem),
        es_mean=("es", "mean"),
        es_sem=("es", sem),
        fair_volatility_mean=("mean_abs_conf_change_per_transition", "mean"),
        fair_volatility_sem=("mean_abs_conf_change_per_transition", sem),
        n=("file", "count"),
    )
    .reset_index()
)

summary.to_csv(OUT_DIR / "summary_fair_volatility_pooled.csv", index=False)
summary_seq.to_csv(OUT_DIR / "summary_fair_volatility_by_sequence.csv", index=False)

print("Saved plots to:", OUT_DIR)
print(summary.round(3))