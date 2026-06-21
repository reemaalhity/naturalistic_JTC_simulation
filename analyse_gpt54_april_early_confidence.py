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

    conf_by_turn = {
        int(t["turn_index"]): float(t["parsed_confidence"])
        for t in turns
        if t.get("turn_index") is not None and t.get("parsed_confidence") is not None
    }

    if 1 not in conf_by_turn or 2 not in conf_by_turn:
        continue

    sequence_file = cond.get("sequence_file", "")
    if "sequence_1" in sequence_file:
        sequence = "sequence_1"
    elif "sequence_2" in sequence_file:
        sequence = "sequence_2"
    else:
        sequence = sequence_file or "unknown"

    rows.append({
        "file": path.name,
        "agent_type": cond.get("agent_type"),
        "reassurance_type": cond.get("reassurance_type"),
        "condition": f"{cond.get('agent_type')} / {cond.get('reassurance_type')}",
        "sequence": sequence,
        "confidence_turn1": conf_by_turn[1],
        "confidence_turn2": conf_by_turn[2],
        "early_confidence_gain": conf_by_turn[2] - conf_by_turn[1],
        "toc_turn": r.get("toc_turn"),
        "accuracy": r.get("accuracy"),
    })

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No April GPT-5.4 runs with turn 1 and turn 2 confidence found.")

df = df.dropna(subset=["toc_turn", "accuracy"]).copy()
df["toc_turn"] = df["toc_turn"].astype(int)
df["accuracy"] = df["accuracy"].astype(int)
df["early_commit"] = df["toc_turn"].eq(2)

df.to_csv(OUT_DIR / "gpt54_april_early_confidence_analysis_raw.csv", index=False)

print("\nMean turn-2 confidence by condition:")
print(df.groupby("condition")["confidence_turn2"].agg(["mean", "count", "std"]))

print("\nEarly confidence gain by condition:")
print(df.groupby("condition")["early_confidence_gain"].agg(["mean", "count", "std"]))

print("\nTurn-2 confidence by accuracy:")
print(df.groupby("accuracy")["confidence_turn2"].agg(["mean", "count", "std"]))

print("\nTurn-2 confidence by early commitment:")
print(df.groupby("early_commit")["confidence_turn2"].agg(["mean", "count", "std"]))

# ---------- Plot 1: turn 2 confidence by condition ----------
summary = df.groupby("condition")["confidence_turn2"].agg(["mean", "count"]).reset_index()

plt.figure(figsize=(8, 5))
plt.bar(summary["condition"], summary["mean"])

for i, row in summary.iterrows():
    plt.text(i, row["mean"] + 1, f"n={int(row['count'])}", ha="center", fontsize=8)

plt.ylim(0, 100)
plt.ylabel("Mean confidence at turn 2")
plt.title("GPT-5.4 April: early confidence by condition")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_april_turn2_confidence_by_condition.png", dpi=300)
plt.close()

# ---------- Plot 2: early confidence gain by condition ----------
gain = df.groupby("condition")["early_confidence_gain"].agg(["mean", "count"]).reset_index()

plt.figure(figsize=(8, 5))
plt.bar(gain["condition"], gain["mean"])

for i, row in gain.iterrows():
    plt.text(i, row["mean"] + 1, f"n={int(row['count'])}", ha="center", fontsize=8)

plt.ylabel("Mean confidence gain from turn 1 to turn 2")
plt.title("GPT-5.4 April: early confidence gain by condition")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_april_early_confidence_gain_by_condition.png", dpi=300)
plt.close()

# ---------- Plot 3: turn 2 confidence by final accuracy ----------
acc = df.groupby("accuracy")["confidence_turn2"].agg(["mean", "count"]).reset_index()
labels = ["Incorrect" if x == 0 else "Correct" for x in acc["accuracy"]]

plt.figure(figsize=(6, 5))
plt.bar(labels, acc["mean"])

for i, row in acc.iterrows():
    plt.text(i, row["mean"] + 1, f"n={int(row['count'])}", ha="center", fontsize=8)

plt.ylim(0, 100)
plt.ylabel("Mean confidence at turn 2")
plt.title("GPT-5.4 April: early confidence by final accuracy")
plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_april_turn2_confidence_by_accuracy.png", dpi=300)
plt.close()

# ---------- Plot 4: JTC only, turn 2 confidence by reassurance + accuracy ----------
jtc = df[df["agent_type"].str.lower() == "jtc"].copy()

jtc_summary = (
    jtc.groupby(["reassurance_type", "accuracy"])["confidence_turn2"]
    .agg(["mean", "count"])
    .reset_index()
)

plt.figure(figsize=(7, 5))

x_positions = []
x_labels = []
values = []
counts = []

for _, row in jtc_summary.iterrows():
    label = f"{row['reassurance_type']}\n{'correct' if row['accuracy'] == 1 else 'incorrect'}"
    x_labels.append(label)
    values.append(row["mean"])
    counts.append(row["count"])

plt.bar(range(len(values)), values)
plt.xticks(range(len(values)), x_labels)
plt.ylim(0, 100)
plt.ylabel("Mean confidence at turn 2")
plt.title("GPT-5.4 April JTC: early confidence by reassurance and accuracy")

for i, (v, n) in enumerate(zip(values, counts)):
    plt.text(i, v + 1, f"n={int(n)}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_april_jtc_turn2_confidence_by_reassurance_accuracy.png", dpi=300)
plt.close()

# ---------- Plot 5: turn 2 confidence vs TOC ----------
plt.figure(figsize=(7, 5))

for condition, sub in df.groupby("condition"):
    plt.scatter(sub["confidence_turn2"], sub["toc_turn"], label=condition, alpha=0.75)

plt.xlabel("Confidence at turn 2")
plt.ylabel("Turn of commitment (TOC)")
plt.title("GPT-5.4 April: early confidence vs commitment timing")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "gpt54_april_turn2_confidence_vs_toc.png", dpi=300)
plt.close()

print("\nSaved figures:")
print(OUT_DIR / "gpt54_april_turn2_confidence_by_condition.png")
print(OUT_DIR / "gpt54_april_early_confidence_gain_by_condition.png")
print(OUT_DIR / "gpt54_april_turn2_confidence_by_accuracy.png")
print(OUT_DIR / "gpt54_april_jtc_turn2_confidence_by_reassurance_accuracy.png")
print(OUT_DIR / "gpt54_april_turn2_confidence_vs_toc.png")
