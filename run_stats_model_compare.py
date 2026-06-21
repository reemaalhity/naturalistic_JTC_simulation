from __future__ import annotations

import json
from pathlib import Path
from math import sqrt

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs_naturalistic_latest"
OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

MAX_EXCERPTS = 6

# ---------------------------------------------------
# LOAD
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
# CLEAN
# ---------------------------------------------------

# Any missing TOC is treated as final-turn commitment
df["toc_plot"] = df["toc_turn"].fillna(MAX_EXCERPTS)

# Keep categories tidy
for col in ["model_name", "agent_type", "reassurance_type", "sequence_name"]:
    df[col] = df[col].astype("category")

# ---------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------

def sem(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) <= 1:
        return 0.0
    return vals.std(ddof=1) / sqrt(len(vals))

summary = (
    df.groupby(
        ["model_name", "agent_type", "reassurance_type", "sequence_name"],
        dropna=False
    )
    .agg(
        mean_toc=("toc_plot", "mean"),
        sd_toc=("toc_plot", "std"),
        sem_toc=("toc_plot", sem),
        mean_evidence=("evidence_requests", "mean"),
        sd_evidence=("evidence_requests", "std"),
        sem_evidence=("evidence_requests", sem),
        mean_accuracy=("accuracy", "mean"),
        sd_accuracy=("accuracy", "std"),
        sem_accuracy=("accuracy", sem),
        n=("file", "count"),
    )
    .reset_index()
)

summary_csv = OUT_DIR / "model_summary_table.csv"
summary.to_csv(summary_csv, index=False)

print(f"Saved summary table: {summary_csv}")

# ---------------------------------------------------
# STATS
# ---------------------------------------------------

# We run per model, per sequence, per measure:
# measure ~ C(agent_type) * C(reassurance_type)

measure_specs = [
    ("evidence_requests", "ES"),
    ("toc_plot", "TOC"),
    ("accuracy", "Accuracy"),
]

all_results = []

for model in df["model_name"].cat.categories:
    model_sub = df[df["model_name"] == model].copy()

    for seq in df["sequence_name"].cat.categories:
        seq_sub = model_sub[model_sub["sequence_name"] == seq].copy()

        if seq_sub.empty:
            continue

        for measure_col, measure_label in measure_specs:
            # Drop rows where measure is missing
            sub = seq_sub.dropna(subset=[measure_col]).copy()

            if len(sub) < 4:
                continue

            try:
                model_fit = smf.ols(
                    f"{measure_col} ~ C(agent_type) * C(reassurance_type)",
                    data=sub
                ).fit()

                anova = sm.stats.anova_lm(model_fit, typ=2)

                for effect_name, row in anova.iterrows():
                    if effect_name == "Residual":
                        continue

                    all_results.append(
                        {
                            "model_name": model,
                            "sequence_name": seq,
                            "measure": measure_label,
                            "effect": effect_name,
                            "sum_sq": row["sum_sq"],
                            "df": row["df"],
                            "F": row["F"],
                            "p_value": row["PR(>F)"],
                            "n": len(sub),
                        }
                    )
            except Exception as e:
                all_results.append(
                    {
                        "model_name": model,
                        "sequence_name": seq,
                        "measure": measure_label,
                        "effect": "MODEL_FAILED",
                        "sum_sq": None,
                        "df": None,
                        "F": None,
                        "p_value": None,
                        "n": len(sub),
                        "error": str(e),
                    }
                )

stats_df = pd.DataFrame(all_results)

stats_csv = OUT_DIR / "model_stats_table.csv"
stats_df.to_csv(stats_csv, index=False)

print(f"Saved stats table: {stats_csv}")
print("Done.")