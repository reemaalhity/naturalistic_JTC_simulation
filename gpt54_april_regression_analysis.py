from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm

RUNS_DIR = Path("runs_latest")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

rows = []

# ---------- LOAD DATA ----------
for path in RUNS_DIR.glob("2026-04-*.json"):

    r = json.load(open(path, encoding="utf-8"))

    if r.get("model_name") != "openai/gpt-5.4":
        continue

    cond = r.get("condition", {})
    turns = r.get("turns", [])

    confs = {
        int(t["turn_index"]): float(t["parsed_confidence"])
        for t in turns
        if t.get("turn_index") is not None
        and t.get("parsed_confidence") is not None
    }

    if 2 not in confs:
        continue

    rows.append({
        "file": path.name,
        "agent_type": cond.get("agent_type"),
        "reassurance_type": cond.get("reassurance_type"),
        "condition": f"{cond.get('agent_type')} / {cond.get('reassurance_type')}",
        "confidence_t2": confs[2],
        "toc": r.get("toc_turn"),
        "accuracy": r.get("accuracy")
    })

df = pd.DataFrame(rows)

df = df.dropna(subset=["confidence_t2", "toc", "accuracy"]).copy()

df["toc"] = df["toc"].astype(float)
df["accuracy"] = df["accuracy"].astype(float)

# ---------- SAVE RAW ----------
df.to_csv(
    OUT_DIR / "gpt54_april_regression_raw.csv",
    index=False
)

# ---------- ALL CONDITIONS ----------
print("\n==============================")
print("ALL CONDITIONS")
print("==============================")

# correlation: confidence vs TOC
r_toc, p_toc = pearsonr(df["confidence_t2"], df["toc"])

print(f"\nConfidence_t2 vs TOC:")
print(f"r = {r_toc:.3f}")
print(f"p = {p_toc:.5f}")

# correlation: confidence vs accuracy
r_acc, p_acc = pearsonr(df["confidence_t2"], df["accuracy"])

print(f"\nConfidence_t2 vs Accuracy:")
print(f"r = {r_acc:.3f}")
print(f"p = {p_acc:.5f}")

# ---------- REGRESSION: TOC ----------
X = sm.add_constant(df["confidence_t2"])
model_toc = sm.OLS(df["toc"], X).fit()

print("\nRegression: TOC ~ confidence_t2")
print(model_toc.summary())

# ---------- REGRESSION: ACCURACY ----------
model_acc = sm.OLS(df["accuracy"], X).fit()

print("\nRegression: Accuracy ~ confidence_t2")
print(model_acc.summary())

# ---------- JTC ONLY ----------
jtc = df[df["agent_type"].str.lower() == "jtc"].copy()

print("\n==============================")
print("JTC ONLY")
print("==============================")

r_toc_jtc, p_toc_jtc = pearsonr(
    jtc["confidence_t2"],
    jtc["toc"]
)

print(f"\nJTC: Confidence_t2 vs TOC")
print(f"r = {r_toc_jtc:.3f}")
print(f"p = {p_toc_jtc:.5f}")

r_acc_jtc, p_acc_jtc = pearsonr(
    jtc["confidence_t2"],
    jtc["accuracy"]
)

print(f"\nJTC: Confidence_t2 vs Accuracy")
print(f"r = {r_acc_jtc:.3f}")
print(f"p = {p_acc_jtc:.5f}")

X_jtc = sm.add_constant(jtc["confidence_t2"])

model_toc_jtc = sm.OLS(
    jtc["toc"],
    X_jtc
).fit()

print("\nJTC Regression: TOC ~ confidence_t2")
print(model_toc_jtc.summary())

model_acc_jtc = sm.OLS(
    jtc["accuracy"],
    X_jtc
).fit()

print("\nJTC Regression: Accuracy ~ confidence_t2")
print(model_acc_jtc.summary())

# ---------- PLOT 1 ----------
plt.figure(figsize=(7, 5))

for condition, sub in df.groupby("condition"):
    plt.scatter(
        sub["confidence_t2"],
        sub["toc"],
        label=condition,
        alpha=0.75
    )

plt.xlabel("Confidence at turn 2")
plt.ylabel("Turn of commitment")
plt.title("GPT-5.4 April: confidence vs TOC")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

plt.savefig(
    OUT_DIR / "gpt54_april_confidence_vs_toc_regression.png",
    dpi=300
)

plt.close()

# ---------- PLOT 2 ----------
plt.figure(figsize=(7, 5))

for acc_value, sub in df.groupby("accuracy"):
    label = "Correct" if acc_value == 1 else "Incorrect"

    plt.scatter(
        sub["confidence_t2"],
        sub["accuracy"],
        label=label,
        alpha=0.75
    )

plt.xlabel("Confidence at turn 2")
plt.ylabel("Accuracy")
plt.title("GPT-5.4 April: confidence vs accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT_DIR / "gpt54_april_confidence_vs_accuracy_regression.png",
    dpi=300
)

plt.close()

print("\nSaved figures:")
print(OUT_DIR / "gpt54_april_confidence_vs_toc_regression.png")
print(OUT_DIR / "gpt54_april_confidence_vs_accuracy_regression.png")

