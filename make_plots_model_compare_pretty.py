from __future__ import annotations

import json
from pathlib import Path
from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs_latest"
OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

MAX_EXCERPTS = 6

# ---------------------------------------------------
# LOAD
# ---------------------------------------------------

rows = []

for path in RUNS_DIR.glob("2026-04-*.json"):
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
    raise ValueError("No JSON files found in runs_latest/")

# ---------------------------------------------------
# CLEAN
# ---------------------------------------------------

df["toc_plot"] = df["toc_turn"].fillna(MAX_EXCERPTS)

model_display_map = {
    "anthropic/claude-opus-4.6": "Claude Opus 4.6",
    "x-ai/grok-4-fast": "Grok-4-fast",
    "openai/gpt-5.4": "GPT-5.4",
}

df["model_display"] = df["model_name"].map(model_display_map).fillna(df["model_name"])

condition_order = [
    ("jtc", "calibrated", "JTC\nCal"),
    ("jtc", "miscalibrated", "JTC\nMiscal"),
    ("nonjtc", "calibrated", "NonJTC\nCal"),
    ("nonjtc", "miscalibrated", "NonJTC\nMiscal"),
]
sequence_order = ["sequence_1", "sequence_2"]

# ---------------------------------------------------
# SUMMARY
# ---------------------------------------------------

def sem(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) <= 1:
        return 0.0
    return vals.std(ddof=1) / sqrt(len(vals))

summary = (
    df.groupby(
        ["model_display", "agent_type", "reassurance_type", "sequence_name"],
        dropna=False
    )
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

models = ["Claude Opus 4.6", "GPT-5.4", "Grok-4-fast"]
models = [m for m in models if m in set(summary["model_display"])]

# ---------------------------------------------------
# PLOT
# ---------------------------------------------------

fig, axes = plt.subplots(
    nrows=3,
    ncols=len(models),
    figsize=(16, 11),
    sharex=False,
    sharey="row",
)

if len(models) == 1:
    axes = axes.reshape(3, 1)

measure_specs = [
    ("mean_evidence", "sem_evidence", "Evidence seeking (ES)", (0, MAX_EXCERPTS)),
    ("mean_toc", "sem_toc", "Turn of commitment (TOC)", (1, MAX_EXCERPTS)),
    ("mean_accuracy", "sem_accuracy", "Accuracy", (0, 1.05)),
]

x_positions = [0, 1, 2, 3]
sequence_offsets = {"sequence_1": -0.08, "sequence_2": 0.08}
marker_map = {"sequence_1": "o", "sequence_2": "s"}
label_map = {"sequence_1": "sequence 1", "sequence_2": "sequence 2"}

for col_idx, model in enumerate(models):
    model_sub = summary[summary["model_display"] == model].copy()

    for row_idx, (mean_col, sem_col, ylabel, ylim) in enumerate(measure_specs):
        ax = axes[row_idx, col_idx]

        for seq in sequence_order:
            means = []
            sems = []

            seq_sub = model_sub[model_sub["sequence_name"] == seq].copy()

            for agent_type, reassurance_type, _label in condition_order:
                cell = seq_sub[
                    (seq_sub["agent_type"] == agent_type)
                    & (seq_sub["reassurance_type"] == reassurance_type)
                ]

                if cell.empty:
                    means.append(float("nan"))
                    sems.append(0.0)
                else:
                    means.append(cell.iloc[0][mean_col])
                    sems.append(cell.iloc[0][sem_col])

            xs = [x + sequence_offsets[seq] for x in x_positions]

            ax.errorbar(
                xs,
                means,
                yerr=sems,
                fmt=marker_map[seq],
                capsize=4,
                markersize=6,
                linewidth=1.2,
                label=label_map[seq] if (row_idx == 0 and col_idx == 0) else None,
            )

        # connect calibrated -> miscalibrated within each agent type
        for seq in sequence_order:
            seq_sub = model_sub[model_sub["sequence_name"] == seq].copy()

            def get_val(a: str, r: str):
                cell = seq_sub[
                    (seq_sub["agent_type"] == a)
                    & (seq_sub["reassurance_type"] == r)
                ]
                if cell.empty:
                    return None
                return cell.iloc[0][mean_col]

            y1 = get_val("jtc", "calibrated")
            y2 = get_val("jtc", "miscalibrated")
            y3 = get_val("nonjtc", "calibrated")
            y4 = get_val("nonjtc", "miscalibrated")

            offset = sequence_offsets[seq]

            if y1 is not None and y2 is not None:
                ax.plot([0 + offset, 1 + offset], [y1, y2], linewidth=1)

            if y3 is not None and y4 is not None:
                ax.plot([2 + offset, 3 + offset], [y3, y4], linewidth=1)

        ax.set_ylim(*ylim)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label for _, _, label in condition_order], fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        if row_idx == 0:
            ax.set_title(model, fontsize=14, pad=8)

        if col_idx == 0:
            ax.set_ylabel(ylabel, fontsize=12)

# overall title + legend
handles, labels = axes[0, 0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11)

fig.suptitle("Model comparison across reassurance conditions", fontsize=16, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_png = OUT_DIR / "model_comparison_pretty.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot: {out_png}")
print("Done.")