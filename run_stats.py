from __future__ import annotations


import json
from pathlib import Path


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs_naturalistic_latest"
OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)


MAX_EXCERPTS = 6
NO_COMMIT_PLOT_VALUE = MAX_EXCERPTS + 1  # plot no-commit as 7


rows = []


for path in sorted(RUNS_DIR.glob("*.json")):
   with path.open("r", encoding="utf-8") as f:
       data = json.load(f)


   cond = data.get("condition", {})
   rows.append(
       {
           "file": path.name,
           "agent_type": cond.get("agent_type"),
           "reassurance_type": cond.get("reassurance_type"),
           "sequence_name": data.get("sequence_name"),
           "accuracy": data.get("accuracy"),
           "evidence_requests": data.get("evidence_requests"),
           "toc_plot": data.get("toc_turn") if data.get("toc_turn") is not None else NO_COMMIT_PLOT_VALUE,
       }
   )


df = pd.DataFrame(rows)


if df.empty:
   raise ValueError("No JSON files found in runs_naturalistic_latest/")


# Clean up categories
df["agent_type"] = df["agent_type"].astype("category")
df["reassurance_type"] = df["reassurance_type"].astype("category")
df["sequence_name"] = df["sequence_name"].astype("category")


measures = ["evidence_requests", "toc_plot", "accuracy"]


all_results = []
raw_pvals = []


for seq in sorted(df["sequence_name"].unique()):
   sub = df[df["sequence_name"] == seq].copy()


   print(f"\n=== Sequence: {seq} ===")


   for measure in measures:
       print(f"\n--- ANOVA for {measure} ---")


       model = smf.ols(
           f"{measure} ~ C(agent_type) * C(reassurance_type)",
           data=sub
       ).fit()


       anova_table = sm.stats.anova_lm(model, typ=2)
       print(anova_table)


       # Save rows for later export
       for effect_name, row in anova_table.iterrows():
           if effect_name == "Residual":
               continue
           pval = row["PR(>F)"]
           raw_pvals.append(pval)
           all_results.append(
               {
                   "sequence_name": seq,
                   "measure": measure,
                   "effect": effect_name,
                   "sum_sq": row["sum_sq"],
                   "df": row["df"],
                   "F": row["F"],
                   "p_uncorrected": pval,
               }
           )


results_df = pd.DataFrame(all_results)


# Benjamini-Hochberg FDR correction across all tested effects
reject, pvals_corrected, _, _ = multipletests(
   results_df["p_uncorrected"],
   method="fdr_bh"
)


results_df["p_fdr_bh"] = pvals_corrected
results_df["reject_fdr_bh"] = reject


out_csv = OUT_DIR / "anova_results.csv"
results_df.to_csv(out_csv, index=False)


print(f"\nSaved ANOVA results to: {out_csv}")
print("\nDone.")
