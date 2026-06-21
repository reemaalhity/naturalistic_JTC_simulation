from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs_latest")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

rows = []

# =====================================================
# LOAD MAY 27 GPT-5.4 RUNS
# =====================================================

for path in RUNS_DIR.glob("2026-05-27*.json"):

    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)

    if r.get("model_name") != "openai/gpt-5.4":
        continue

    cond = r.get("condition", {})
    turns = r.get("turns", [])

    confs = []
    turns_seen = []

    for t in turns:
        conf = t.get("parsed_confidence")
        turn_index = t.get("turn_index")

        if conf is None or turn_index is None:
            continue

        confs.append(float(conf))
        turns_seen.append(int(turn_index))

    if len(confs) < 2:
        continue

    diffs = [abs(confs[i] - confs[i - 1]) for i in range(1, len(confs))]

    rows.append({
        "file": path.name,
        "agent_type": cond.get("agent_type"),
        "reassurance_type": cond.get("reassurance_type"),
        "reassurance_delivery_style": cond.get(
            "reassurance_delivery_style",
            "legacy_binary"
        ),
        "sequence_file": cond.get("sequence_file"),
        "toc_turn": r.get("toc_turn"),
        "evidence_requests": r.get("evidence_requests"),
        "accuracy": r.get("accuracy"),
        "final_confidence": r.get("final_confidence"),

        # volatility metrics
        "mean_abs_confidence_change": np.mean(diffs),
        "max_abs_confidence_jump": np.max(diffs),
        "sd_confidence": np.std(confs, ddof=1),
        "confidence_delta_turn_1_to_2": confs[1] - confs[0],
        "confidence_range": max(confs) - min(confs),
        "n_confidence_points": len(confs),
    })

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No usable May 27 GPT-5.4 runs found.")

df["condition"] = (
    df["agent_type"]
    + " / "
    + df["reassurance_type"]
    + " / "
    + df["reassurance_delivery_style"]
)

df.to_csv(
    OUT_DIR / "may27_confidence_volatility_toc_es_raw.csv",
    index=False
)

# =====================================================
# SUMMARY TABLE
# =====================================================

summary = (
    df.groupby(["agent_type", "reassurance_type", "reassurance_delivery_style"])
    .agg(
        n=("file", "count"),
        mean_volatility=("mean_abs_confidence_change", "mean"),
        sem_volatility=("mean_abs_confidence_change", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        mean_toc=("toc_turn", "mean"),
        sem_toc=("toc_turn", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        mean_es=("evidence_requests", "mean"),
        sem_es=("evidence_requests", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        mean_accuracy=("accuracy", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    OUT_DIR / "may27_confidence_volatility_toc_es_summary.csv",
    index=False
)

# =====================================================
# CORRELATION TABLE
# =====================================================

corr_vars = [
    "mean_abs_confidence_change",
    "max_abs_confidence_jump",
    "sd_confidence",
    "confidence_delta_turn_1_to_2",
    "confidence_range",
    "toc_turn",
    "evidence_requests",
    "accuracy",
]

corr = df[corr_vars].corr(numeric_only=True)

corr.to_csv(
    OUT_DIR / "may27_confidence_volatility_toc_es_correlations.csv"
)

# =====================================================
# PLOT 1: VOLATILITY × TOC
# =====================================================

plt.figure(figsize=(8, 6))

for condition, sub in df.groupby("condition"):
    plt.scatter(
        sub["toc_turn"],
        sub["mean_abs_confidence_change"],
        alpha=0.75,
        label=condition
    )

plt.xlabel("Time to commitment (TOC turn)")
plt.ylabel("Mean absolute confidence change")
plt.title("Confidence volatility × TOC")
plt.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.savefig(
    OUT_DIR / "may27_volatility_by_toc_scatter.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =====================================================
# PLOT 2: VOLATILITY × EVIDENCE SAMPLING
# =====================================================

plt.figure(figsize=(8, 6))

for condition, sub in df.groupby("condition"):
    plt.scatter(
        sub["evidence_requests"],
        sub["mean_abs_confidence_change"],
        alpha=0.75,
        label=condition
    )

plt.xlabel("Evidence requests (ES)")
plt.ylabel("Mean absolute confidence change")
plt.title("Confidence volatility × evidence sampling")
plt.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.savefig(
    OUT_DIR / "may27_volatility_by_es_scatter.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =====================================================
# PLOT 3: COMPACT CONDITION SUMMARY
# =====================================================

plot_summary = (
    df.groupby("condition")
    .agg(
        mean_volatility=("mean_abs_confidence_change", "mean"),
        sem_volatility=("mean_abs_confidence_change", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        mean_toc=("toc_turn", "mean"),
        sem_toc=("toc_turn", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        mean_es=("evidence_requests", "mean"),
        sem_es=("evidence_requests", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        n=("file", "count"),
    )
    .reset_index()
    .sort_values("condition")
)

metrics = [
    ("mean_volatility", "sem_volatility", "Confidence volatility"),
    ("mean_toc", "sem_toc", "TOC"),
    ("mean_es", "sem_es", "Evidence sampling"),
]

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 9),
    sharex=True
)

for ax, (mean_col, sem_col, label) in zip(axes, metrics):

    x = np.arange(len(plot_summary))

    ax.errorbar(
        x,
        plot_summary[mean_col],
        yerr=plot_summary[sem_col],
        fmt="o",
        capsize=4,
        linewidth=1.5
    )

    ax.plot(
        x,
        plot_summary[mean_col],
        linewidth=1,
        alpha=0.5
    )

    ax.set_ylabel(label)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

axes[-1].set_xticks(np.arange(len(plot_summary)))
axes[-1].set_xticklabels(
    plot_summary["condition"],
    rotation=45,
    ha="right"
)

plt.suptitle("Confidence volatility, TOC, and evidence sampling by condition")
plt.tight_layout()

plt.savefig(
    OUT_DIR / "may27_volatility_toc_es_condition_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =====================================================
# PLOT 4: CORRELATION HEATMAP
# =====================================================

plt.figure(figsize=(8, 7))

im = plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)

plt.colorbar(im, label="Pearson r")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation matrix: confidence volatility, TOC, ES")

for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        val = corr.iloc[i, j]
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "may27_volatility_toc_es_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nSaved:")
print(OUT_DIR / "may27_confidence_volatility_toc_es_raw.csv")
print(OUT_DIR / "may27_confidence_volatility_toc_es_summary.csv")
print(OUT_DIR / "may27_confidence_volatility_toc_es_correlations.csv")
print(OUT_DIR / "may27_volatility_by_toc_scatter.png")
print(OUT_DIR / "may27_volatility_by_es_scatter.png")
print(OUT_DIR / "may27_volatility_toc_es_condition_summary.png")
print(OUT_DIR / "may27_volatility_toc_es_correlation_heatmap.png")