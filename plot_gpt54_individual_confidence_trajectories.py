from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs_latest")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

rows = []

for path in RUNS_DIR.glob("2026-04-*.json"):

    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)

    if r.get("model_name") != "openai/gpt-5.4":
        continue

    cond = r.get("condition", {})
    turns = r.get("turns", [])

    condition = (
        cond.get("agent_type", "") +
        " / " +
        cond.get("reassurance_type", "")
    )

    for t in turns:

        conf = t.get("parsed_confidence")
        turn_index = t.get("turn_index")

        if conf is None or turn_index is None:
            continue

        rows.append({
            "file": path.name,
            "condition": condition,
            "turn_index": int(turn_index),
            "confidence": float(conf)
        })

df = pd.DataFrame(rows)

conditions = df["condition"].unique()

fig, axes = plt.subplots(
    len(conditions),
    1,
    figsize=(8, 4 * len(conditions)),
    sharex=True
)

if len(conditions) == 1:
    axes = [axes]

for ax, condition in zip(axes, conditions):

    sub = df[df["condition"] == condition]

    # plot individual runs faintly
    for file, run_df in sub.groupby("file"):

        run_df = run_df.sort_values("turn_index")

        ax.plot(
            run_df["turn_index"],
            run_df["confidence"],
            alpha=0.2,
            linewidth=1
        )

    # plot mean trajectory bold
    mean_df = (
        sub.groupby("turn_index")["confidence"]
        .mean()
        .reset_index()
        .sort_values("turn_index")
    )

    ax.plot(
        mean_df["turn_index"],
        mean_df["confidence"],
        linewidth=3,
        marker="o"
    )

    ax.set_title(condition)
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 100)

axes[-1].set_xlabel("Turn index")

plt.suptitle(
    "GPT-5.4 April runs: individual confidence trajectories",
    y=0.995
)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "gpt54_individual_confidence_trajectories.png",
    dpi=300
)

print("\nSaved:")
print(OUT_DIR / "gpt54_individual_confidence_trajectories.png")
