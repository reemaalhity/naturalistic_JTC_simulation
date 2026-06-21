from __future__ import annotations

import json
from pathlib import Path
from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs_naturalistic_latest"
OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

MAX_EXCERPTS = 6

# ---------------------------------------------------
# LOAD JSON RUNS
# ---------------------------------------------------

rows = []

for path in sorted(RUNS_DIR.glob("*.json")):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cond = data.get("condition", {})

    rows.append(
        {
            "file": path.name,
            "model_name": data.get("model_name"),
            "agent_type": cond.get("agent_type"),
            "reassurance_type": cond.get("reassurance_type"),
            "sequence_name": data.get("sequence_name"),
            "accuracy": data.get("accuracy"),
            "evidence_requests": data.get("evidence_requests"),
            "toc_turn": data.get("toc_turn"),
        }
    )

df = pd.DataFrame(rows)

if df.empty:
    raise ValueError("No JSON files found in runs_naturalistic_latest/")

# ---------------------------------------------------
# CLEANING
# ---------------------------------------------------

# Force final-turn commits to remain as 6, not missing
df["toc_plot"] = df["toc_turn"].fillna(MAX_EXCERPTS)

df["condition_label"] = (
    df["agent_type"].map({"jtc": "JTC", "nonjtc": "NonJTC"})
    + "\n"
    + df["reassurance_type"].map({"calibrated": "Cal", "miscalibrated": "Miscal"})
)

condition_order = ["JTC\nCal", "JTC\nMiscal", "NonJTC\nCal", "NonJTC\nMiscal"]
sequence_order = ["sequence_1", "sequence_2"]

# Optional nicer display names
model_display_map = {
    "openai/gpt-5.4": "GPT-5.4",
}

df["model_display"] = df["model_name"].map(model_display_map).fillna(df["model_name"])

# ---------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------

def sem(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) <= 1:
        return 0.0
    return vals.std(ddof=1) / sqrt(len(vals))

summary = (
    df.groupby(["agent_type", "reassurance_type", "sequence_name"], dropna=False)
      .agg(
          mean_toc=("toc_plot", "mean"),
          sem_toc=("toc_plot", sem),
          mean_evidence=("evidence_requests", "mean"),
          sem_evidence=("evidence_requests", sem),
          mean_accuracy=("accuracy", "mean"),
          sem_accuracy=("accuracy", sem),
          n=("file", "count"),
      )
      .reset_index()
)

summary_csv = OUT_DIR / "model_summary_table.csv"
summary.to_csv(summary_csv, index=False)

print(f"Saved summary table: {summary_csv}")

# ---------------------------------------------------
# PLOTTING
# ---------------------------------------------------

models = list(summary["model_display"].dropna().unique())

if not models:
    raise ValueError("No model names found.")

fig, axes = plt.subplots(
    nrows=3,
    ncols=len(models),
    figsize=(5 * len(models), 11),
    sharex=False,
    sharey="row",
)

if len(models) == 1:
    # Make axes consistently 2D
    axes = axes.reshape(3, 1)

measure_specs = [
    ("mean_evidence", "sem_evidence", "Evidence seeking (ES)", (0, MAX_EXCERPTS)),
    ("mean_toc", "sem_toc", "Turn of commitment (TOC)", (1, MAX_EXCERPTS)),
    ("mean_accuracy", "sem_accuracy", "Accuracy", (0, 1.05)),
]

x_positions = [0, 1, 2, 3]
offset_map = {"sequence_1": -0.08, "sequence_2": 0.08}
marker_map = {"sequence_1": "o", "sequence_2": "s"}
label_map = {"sequence_1": "sequence 1", "sequence_2": "sequence 2"}

for col_idx, model in enumerate(models):
    model_sub = summary[summary["model_display"] == model].copy()

    for row_idx, (mean_col, sem_col, ylab, ylim) in enumerate(measure_specs):
        ax = axes[row_idx, col_idx]

        for seq in sequence_order:
            seq_sub = model_sub[model_sub["sequence_name"] == seq].copy()

            means = []
            sems = []

            for cond in condition_order:
                agent_part, reass_part = cond.split("\n")
                agent_lookup = "jtc" if agent_part == "JTC" else "nonjtc"
                reass_lookup = "calibrated" if reass_part == "Cal" else "miscalibrated"

                cell = seq_sub[
                    (seq_sub["agent_type"] == agent_lookup)
                    & (seq_sub["reassurance_type"] == reass_lookup)
                ]

                if cell.empty:
                    means.append(float("nan"))
                    sems.append(0.0)
                else:
                    means.append(cell.iloc[0][mean_col])
                    sems.append(cell.iloc[0][sem_col])

            xs = [x + offset_map[seq] for x in x_positions]

            ax.errorbar(
                xs,
                means,
                yerr=sems,
                fmt=marker_map[seq],
                capsize=4,
                label=label_map[seq] if row_idx == 0 else None,
            )

        ax.set_title(model if row_idx == 0 else "")
        ax.set_ylabel(ylab if col_idx == 0 else "")
        ax.set_ylim(*ylim)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(condition_order)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

# Shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

fig.suptitle("Model comparison across reassurance conditions", y=0.98, fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_png = OUT_DIR / "model_comparison_figure.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot: {out_png}")
print("Done.")