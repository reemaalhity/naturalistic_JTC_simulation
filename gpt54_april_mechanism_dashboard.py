from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs_latest")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

rows = []

for path in RUNS_DIR.glob("2026-04-*.json"):
    r = json.load(open(path, encoding="utf-8"))
    if r.get("model_name") != "openai/gpt-5.4":
        continue

    cond = r.get("condition", {})
    turns = r.get("turns", [])

    confs = []
    for t in turns:
        if t.get("turn_index") is not None and t.get("parsed_confidence") is not None:
            confs.append((int(t["turn_index"]), float(t["parsed_confidence"])))

    if len(confs) < 2:
        continue

    confs = sorted(confs)
    conf_dict = dict(confs)
    diffs = [abs(confs[i][1] - confs[i-1][1]) for i in range(1, len(confs))]

    seq_file = cond.get("sequence_file", "")
    sequence = "sequence_1" if "sequence_1" in seq_file else "sequence_2"

    rows.append({
        "file": path.name,
        "condition": f"{cond.get('agent_type')} / {cond.get('reassurance_type')}",
        "agent_type": cond.get("agent_type"),
        "reassurance_type": cond.get("reassurance_type"),
        "sequence": sequence,
        "confidence_t1": conf_dict.get(1),
        "confidence_t2": conf_dict.get(2),
        "early_gain": conf_dict.get(2) - conf_dict.get(1) if 1 in conf_dict and 2 in conf_dict else None,
        "volatility": sum(diffs) / len(diffs),
        "toc": r.get("toc_turn"),
        "accuracy": r.get("accuracy"),
    })

df = pd.DataFrame(rows).dropna(subset=["confidence_t2", "early_gain", "volatility", "toc", "accuracy"])
df["toc"] = df["toc"].astype(int)
df["accuracy"] = df["accuracy"].astype(int)
df.to_csv(OUT_DIR / "gpt54_april_mechanism_dashboard_data.csv", index=False)

order = ["jtc / calibrated", "jtc / miscalibrated", "nonjtc / calibrated", "nonjtc / miscalibrated"]
df["condition"] = pd.Categorical(df["condition"], categories=order, ordered=True)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 1. confidence trajectory
ax = axes[0, 0]
for condition, sub in df.groupby("condition", observed=True):
    traj_rows = []
    for file in sub["file"].unique():
        r = json.load(open(RUNS_DIR / file, encoding="utf-8"))
        for t in r.get("turns", []):
            if t.get("turn_index") is not None and t.get("parsed_confidence") is not None:
                traj_rows.append({
                    "condition": condition,
                    "turn": int(t["turn_index"]),
                    "confidence": float(t["parsed_confidence"])
                })
    traj = pd.DataFrame(traj_rows)
    mean_traj = traj.groupby("turn")["confidence"].mean().reset_index()
    ax.plot(mean_traj["turn"], mean_traj["confidence"], marker="o", label=condition)
ax.set_title("Mean confidence trajectory")
ax.set_xlabel("Turn")
ax.set_ylabel("Confidence")
ax.set_ylim(0, 100)
ax.legend(fontsize=8)

# 2. turn 2 confidence
ax = axes[0, 1]
summary = df.groupby("condition", observed=True)["confidence_t2"].mean()
ax.bar(summary.index.astype(str), summary.values)
ax.set_title("Early confidence at turn 2")
ax.set_ylabel("Confidence")
ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=25)

# 3. volatility
ax = axes[0, 2]
summary = df.groupby("condition", observed=True)["volatility"].mean()
ax.bar(summary.index.astype(str), summary.values)
ax.set_title("Confidence volatility")
ax.set_ylabel("Mean absolute confidence change")
ax.tick_params(axis="x", rotation=25)

# 4. TOC
ax = axes[1, 0]
summary = df.groupby("condition", observed=True)["toc"].mean()
ax.bar(summary.index.astype(str), summary.values)
ax.set_title("Turn of commitment")
ax.set_ylabel("Mean TOC")
ax.set_ylim(1, max(6, df["toc"].max() + 0.5))
ax.tick_params(axis="x", rotation=25)

# 5. accuracy
ax = axes[1, 1]
summary = df.groupby("condition", observed=True)["accuracy"].mean()
ax.bar(summary.index.astype(str), summary.values)
ax.set_title("Final accuracy")
ax.set_ylabel("Mean accuracy")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=25)

# 6. early confidence vs TOC/accuracy
ax = axes[1, 2]
for acc, sub in df.groupby("accuracy"):
    label = "Correct" if acc == 1 else "Incorrect"
    ax.scatter(sub["confidence_t2"], sub["toc"], alpha=0.75, label=label)
ax.set_title("Early confidence vs commitment")
ax.set_xlabel("Confidence at turn 2")
ax.set_ylabel("TOC")
ax.legend()

plt.suptitle("GPT-5.4 April mechanism dashboard", fontsize=16)
plt.tight_layout()
out = OUT_DIR / "gpt54_april_mechanism_dashboard.png"
plt.savefig(out, dpi=300)
plt.close()

print("Saved:", out)
print("Saved data:", OUT_DIR / "gpt54_april_mechanism_dashboard_data.csv")
