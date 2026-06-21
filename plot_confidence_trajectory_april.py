from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs_latest")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

MODEL_FILTER = None  # set to "openai/gpt-5.4" if you only want GPT-5.4

rows = []

for path in RUNS_DIR.glob("2026-04-*.json"):
    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)

    model = r.get("model_name", "")

    if MODEL_FILTER is not None and model != MODEL_FILTER:
        continue

    cond = r.get("condition", {})
    turns = r.get("turns", [])

    agent_type = cond.get("agent_type")
    reassurance_type = cond.get("reassurance_type")
    sequence_file = cond.get("sequence_file")

    sequence = "unknown"
    if sequence_file:
        if "sequence_1" in sequence_file:
            sequence = "sequence_1"
        elif "sequence_2" in sequence_file:
            sequence = "sequence_2"
        else:
            sequence = sequence_file

    for t in turns:
        conf = t.get("parsed_confidence")
        turn_index = t.get("turn_index")

        if conf is None or turn_index is None:
            continue

        rows.append({
            "file": path.name,
            "model": model,
            "agent_type": agent_type,
            "reassurance_type": reassurance_type,
            "sequence": sequence,
            "turn_index": int(turn_index),
            "confidence": float(conf),
            "accuracy": r.get("accuracy"),
            "toc_turn": r.get("toc_turn"),
        })

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No confidence trajectory data found in April runs.")

df["condition"] = df["agent_type"] + " / " + df["reassurance_type"]

# Save raw long-form data
df.to_csv(OUT_DIR / "april_confidence_trajectory_raw.csv", index=False)

# ---------- Plot 1: JTC confidence trajectory, calibrated vs miscalibrated ----------
jtc = df[df["agent_type"].str.lower() == "jtc"].copy()

summary_jtc = (
    jtc.groupby(["reassurance_type", "turn_index"])["confidence"]
    .agg(["mean", "count", "std"])
    .reset_index()
)

summary_jtc.to_csv(OUT_DIR / "april_jtc_confidence_trajectory_summary.csv", index=False)

plt.figure(figsize=(7, 5))

for reassurance, sub in summary_jtc.groupby("reassurance_type"):
    sub = sub.sort_values("turn_index")
    plt.plot(
        sub["turn_index"],
        sub["mean"],
        marker="o",
        linewidth=2,
        label=reassurance
    )

    for _, row in sub.iterrows():
        plt.text(
            row["turn_index"],
            row["mean"] + 1,
            f"n={int(row['count'])}",
            ha="center",
            fontsize=8
        )

plt.ylim(0, 100)
plt.xlabel("Turn index")
plt.ylabel("Mean parsed confidence")
plt.title("April runs: JTC confidence trajectory")
plt.legend(title="Reassurance type")
plt.tight_layout()
plt.savefig(OUT_DIR / "april_jtc_confidence_trajectory.png", dpi=300)
plt.close()

# ---------- Plot 2: confidence trajectory by agent + reassurance ----------
summary_all = (
    df.groupby(["condition", "turn_index"])["confidence"]
    .agg(["mean", "count", "std"])
    .reset_index()
)

summary_all.to_csv(OUT_DIR / "april_all_conditions_confidence_trajectory_summary.csv", index=False)

conditions = summary_all["condition"].unique()

fig, axes = plt.subplots(
    len(conditions),
    1,
    figsize=(8, 4 * len(conditions)),
    sharex=True
)

if len(conditions) == 1:
    axes = [axes]

for ax, condition in zip(axes, conditions):
    sub = summary_all[summary_all["condition"] == condition].sort_values("turn_index")

    ax.plot(sub["turn_index"], sub["mean"], marker="o", linewidth=2)

    for _, row in sub.iterrows():
        ax.text(
            row["turn_index"],
            row["mean"] + 1,
            f"n={int(row['count'])}",
            ha="center",
            fontsize=8
        )

    ax.set_title(condition)
    ax.set_ylabel("Mean confidence")
    ax.set_ylim(0, 100)

axes[-1].set_xlabel("Turn index")

plt.suptitle("April runs: confidence trajectory by condition", y=0.995)
plt.tight_layout()
plt.savefig(OUT_DIR / "april_all_conditions_confidence_trajectory.png", dpi=300)
plt.close()

# ---------- Volatility analysis ----------
# Mean absolute confidence change per run
vol_rows = []

for file, sub in df.groupby("file"):
    sub = sub.sort_values("turn_index")
    confs = sub["confidence"].tolist()

    if len(confs) < 2:
        continue

    diffs = [abs(confs[i] - confs[i-1]) for i in range(1, len(confs))]

    first = sub.iloc[0]

    vol_rows.append({
        "file": file,
        "model": first["model"],
        "agent_type": first["agent_type"],
        "reassurance_type": first["reassurance_type"],
        "sequence": first["sequence"],
        "mean_abs_confidence_change": sum(diffs) / len(diffs),
        "early_confidence_gain": confs[1] - confs[0],
        "accuracy": first["accuracy"],
        "toc_turn": first["toc_turn"],
    })

vol = pd.DataFrame(vol_rows)
vol.to_csv(OUT_DIR / "april_confidence_volatility_by_run.csv", index=False)

vol_summary = (
    vol.groupby(["agent_type", "reassurance_type"])["mean_abs_confidence_change"]
    .agg(["mean", "count", "std"])
    .reset_index()
)

vol_summary.to_csv(OUT_DIR / "april_confidence_volatility_summary.csv", index=False)

plt.figure(figsize=(7, 5))

labels = []
values = []
counts = []

for _, row in vol_summary.iterrows():
    labels.append(f"{row['agent_type']}\n{row['reassurance_type']}")
    values.append(row["mean"])
    counts.append(row["count"])

plt.bar(labels, values)

for i, (v, n) in enumerate(zip(values, counts)):
    plt.text(i, v + 0.3, f"n={int(n)}", ha="center", fontsize=8)

plt.ylabel("Mean absolute confidence change")
plt.title("April runs: confidence volatility by condition")
plt.tight_layout()
plt.savefig(OUT_DIR / "april_confidence_volatility_by_condition.png", dpi=300)
plt.close()

# ---------- Early confidence gain ----------
gain_summary = (
    vol.groupby(["agent_type", "reassurance_type"])["early_confidence_gain"]
    .agg(["mean", "count", "std"])
    .reset_index()
)

gain_summary.to_csv(OUT_DIR / "april_early_confidence_gain_summary.csv", index=False)

plt.figure(figsize=(7, 5))

labels = []
values = []
counts = []

for _, row in gain_summary.iterrows():
    labels.append(f"{row['agent_type']}\n{row['reassurance_type']}")
    values.append(row["mean"])
    counts.append(row["count"])

plt.bar(labels, values)

for i, (v, n) in enumerate(zip(values, counts)):
    plt.text(i, v + 0.3, f"n={int(n)}", ha="center", fontsize=8)

plt.ylabel("Confidence change from turn 1 to turn 2")
plt.title("April runs: early confidence gain by condition")
plt.tight_layout()
plt.savefig(OUT_DIR / "april_early_confidence_gain_by_condition.png", dpi=300)
plt.close()

print("\nSaved figures:")
print(OUT_DIR / "april_jtc_confidence_trajectory.png")
print(OUT_DIR / "april_all_conditions_confidence_trajectory.png")
print(OUT_DIR / "april_confidence_volatility_by_condition.png")
print(OUT_DIR / "april_early_confidence_gain_by_condition.png")

print("\nSaved CSVs:")
print(OUT_DIR / "april_confidence_trajectory_raw.csv")
print(OUT_DIR / "april_confidence_volatility_by_run.csv")
print(OUT_DIR / "april_jtc_confidence_trajectory_summary.csv")
