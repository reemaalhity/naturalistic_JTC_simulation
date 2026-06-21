# figure1_naturalistic.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

RUN_DIR = "test_run_naturalistic"
OUTFILE = "figure1_naturalistic.png"

# =====================================================
# LOAD DATA
# =====================================================

rows = []

for f in Path(RUN_DIR).glob("*.json"):

    with open(f, "r") as fp:
        d = json.load(fp)

    cond = d["condition"]

    rows.append({
        "agent": cond["agent_type"],
        "reassurance": cond["reassurance_type"],
        "sequence": cond["sequence_file"],
        "toc": d["toc_turn"],
        "es": d["evidence_requests"],
        "accuracy": d["accuracy"],
    })

df = pd.DataFrame(rows)

# -----------------------------------------------------
# sequence labels
# -----------------------------------------------------

df["sequence"] = (
    df["sequence"]
    .str.extract(r"(sequence_\d+)")
)

# =====================================================
# SUMMARY
# =====================================================

summary = (
    df.groupby(
        ["agent", "reassurance", "sequence"],
        dropna=False
    )
    .agg(
        es_mean=("es", "mean"),
        es_sem=("es", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        toc_mean=("toc", "mean"),
        toc_sem=("toc", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        acc_mean=("accuracy", "mean"),
        acc_sem=("accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    )
    .reset_index()
)

print(summary)

# =====================================================
# PLOTTING HELPERS
# =====================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(8, 10),
    sharex=True
)

metrics = [
    ("es_mean", "es_sem", "ES"),
    ("toc_mean", "toc_sem", "TOC"),
    ("acc_mean", "acc_sem", "Accuracy"),
]

colors = {
    "sequence_1": "#1f77b4",
    "sequence_2": "#d62728",
}

xpos = {
    ("jtc", "calibrated"): 1,
    ("jtc", "miscalibrated"): 2,
    ("nonjtc", "calibrated"): 4,
    ("nonjtc", "miscalibrated"): 5,
}

# =====================================================
# MAIN PANELS
# =====================================================

for ax, (metric, sem_metric, ylabel) in zip(axes, metrics):

    for _, row in summary.iterrows():

        x = xpos[(row.agent, row.reassurance)]

        offset = -0.08 if row.sequence == "sequence_1" else 0.08

        ax.errorbar(
            x + offset,
            row[metric],
            yerr=row[sem_metric],
            fmt="o",
            color=colors[row.sequence],
            markersize=8,
            capsize=4,
        )

    ax.set_ylabel(ylabel)

    ax.axvline(
        3,
        linestyle="--",
        color="gray",
        alpha=.5,
    )

# =====================================================
# LABELS
# =====================================================

axes[0].set_title(
    "Main results across both sequences",
    fontsize=16,
    pad=15
)

axes[-1].set_xticks([1, 2, 4, 5])

axes[-1].set_xticklabels([
    "JTC\nCal",
    "JTC\nMiscal",
    "NonJTC\nCal",
    "NonJTC\nMiscal",
])

# =====================================================
# LEGEND
# =====================================================

from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="sequence 1",
        markerfacecolor=colors["sequence_1"],
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="sequence 2",
        markerfacecolor=colors["sequence_2"],
        markersize=10,
    ),
]

axes[0].legend(
    handles=legend_elements,
    frameon=False,
    loc="upper left",
)

plt.tight_layout()

plt.savefig(
    OUTFILE,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nSaved -> {OUTFILE}")