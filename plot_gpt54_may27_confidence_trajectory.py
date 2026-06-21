from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs_latest")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

rows = []

# ---------- Load May 27th GPT-5.4 runs only ----------
for path in RUNS_DIR.glob("2026-05-27*.json"):

    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)

    if r.get("model_name") != "openai/gpt-5.4":
        continue

    cond = r.get("condition", {})
    turns = r.get("turns", [])

    agent_type = cond.get("agent_type")
    reassurance_type = cond.get("reassurance_type")
    reassurance_delivery_style = cond.get(
        "reassurance_delivery_style",
        "legacy_binary"
    )

    sequence_file = cond.get("sequence_file", "")
    sequence = "sequence_1" if "sequence_1" in sequence_file else "sequence_2"

    for t in turns:
        conf = t.get("parsed_confidence")
        turn_index = t.get("turn_index")

        if conf is None or turn_index is None:
            continue

        rows.append({
            "file": path.name,
            "agent_type": agent_type,
            "reassurance_type": reassurance_type,
            "reassurance_delivery_style": reassurance_delivery_style,
            "sequence": sequence,
            "turn_index": int(turn_index),
            "confidence": float(conf),
            "accuracy": r.get("accuracy"),
            "toc_turn": r.get("toc_turn"),
        })

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No May27 GPT-5.4 runs found.")

df["condition"] = df["agent_type"] + " / " + df["reassurance_type"]

# ---------- Save raw ----------
df.to_csv(
    OUT_DIR / "gpt54_May27_confidence_trajectory_raw_by_delivery_style.csv",
    index=False
)

# ---------- Trajectory plots: one figure per delivery style ----------
saved_trajectory_paths = []

for style in sorted(df["reassurance_delivery_style"].unique()):

    style_df = df[df["reassurance_delivery_style"] == style].copy()

    style_summary = (
        style_df.groupby(
            ["agent_type", "reassurance_type", "turn_index"]
        )["confidence"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for (agent, reassurance), sub in style_summary.groupby(
        ["agent_type", "reassurance_type"]
    ):

        sub = sub.sort_values("turn_index")

        plt.plot(
            sub["turn_index"],
            sub["mean"],
            marker="o",
            linewidth=2,
            label=f"{agent} / {reassurance}"
        )

    plt.ylim(0, 100)
    plt.xlabel("Turn index")
    plt.ylabel("Mean parsed confidence")
    plt.title(f"Confidence trajectory: {style}")
    plt.legend(title="Condition", fontsize=8)
    plt.tight_layout()

    safe_style = style.replace(" ", "_").replace("/", "_")

    out_path = OUT_DIR / f"gpt54_May27_confidence_trajectory_{safe_style}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    saved_trajectory_paths.append(out_path)

# ---------- Volatility analysis ----------
vol_rows = []

for file, sub in df.groupby("file"):

    sub = sub.sort_values("turn_index")
    confs = sub["confidence"].tolist()

    if len(confs) < 2:
        continue

    diffs = [
        abs(confs[i] - confs[i - 1])
        for i in range(1, len(confs))
    ]

    first = sub.iloc[0]

    vol_rows.append({
        "file": file,
        "agent_type": first["agent_type"],
        "reassurance_type": first["reassurance_type"],
        "reassurance_delivery_style": first["reassurance_delivery_style"],
        "condition": first["condition"],
        "mean_abs_confidence_change": sum(diffs) / len(diffs),
        "max_abs_confidence_jump": max(diffs),
        "early_confidence_gain": confs[1] - confs[0],
        "accuracy": first["accuracy"],
    })

vol = pd.DataFrame(vol_rows)

if vol.empty:
    raise SystemExit("No runs with at least two confidence values found.")

vol.to_csv(
    OUT_DIR / "gpt54_May27_confidence_volatility_raw_by_delivery_style.csv",
    index=False
)

# =====================================================
# COMPACT ABSTRACT-STYLE SUMMARY FIGURE
# =====================================================

import numpy as np

metrics = [
    ("mean_abs_confidence_change", "Volatility"),
    ("early_confidence_gain", "Early gain"),
]

styles = sorted(vol["reassurance_delivery_style"].unique())

summary_rows = []

for metric, _ in metrics:

    grouped = (
        vol.groupby([
            "agent_type",
            "reassurance_type",
            "reassurance_delivery_style"
        ])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    grouped["sem"] = (
        grouped["std"] / np.sqrt(grouped["count"])
    )

    grouped["metric"] = metric

    summary_rows.append(grouped)

summary_df = pd.concat(summary_rows, ignore_index=True)

# ---------- X positions ----------
x_positions = {
    ("jtc", "calibrated"): 0,
    ("jtc", "miscalibrated"): 1,
    ("nonjtc", "calibrated"): 3,
    ("nonjtc", "miscalibrated"): 4,
}

style_offsets = {
    style: offset
    for style, offset in zip(styles, [-0.18, -0.06, 0.06, 0.18])
}

# ---------- Figure ----------
fig, axes = plt.subplots(
    len(metrics),
    1,
    figsize=(8, 6),
    sharex=True
)

if len(metrics) == 1:
    axes = [axes]

for ax, (metric, label) in zip(axes, metrics):

    metric_df = summary_df[
        summary_df["metric"] == metric
    ]

    for style in styles:

        sub = metric_df[
            metric_df["reassurance_delivery_style"] == style
        ]

        xs = []
        ys = []
        sems = []

        for _, row in sub.iterrows():

            base_x = x_positions[
                (row["agent_type"], row["reassurance_type"])
            ]

            xs.append(
                base_x + style_offsets[style]
            )

            ys.append(row["mean"])
            sems.append(row["sem"])

        ax.errorbar(
            xs,
            ys,
            yerr=sems,
            fmt="o",
            capsize=4,
            markersize=7,
            linewidth=1.5,
            label=style
        )

    ax.set_ylabel(label)

    ax.axvline(
        2,
        linestyle="--",
        linewidth=1,
        alpha=0.4
    )

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.3
    )

# ---------- X labels ----------
axes[-1].set_xticks([0, 1, 3, 4])

axes[-1].set_xticklabels([
    "JTC\nCal",
    "JTC\nMiscal",
    "NonJTC\nCal",
    "NonJTC\nMiscal",
])

# ---------- Legend ----------
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False
)

plt.suptitle(
    "Effects of reassurance delivery style on confidence dynamics",
    y=0.98
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.savefig(
    OUT_DIR / "abstract_style_confidence_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    OUT_DIR / "abstract_style_confidence_summary.png"
)