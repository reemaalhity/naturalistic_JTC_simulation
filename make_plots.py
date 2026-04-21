from __future__ import annotations


import json
from pathlib import Path
from typing import Any


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs_latest"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


MAX_EXCERPTS = 6
NO_COMMIT_PLOT_VALUE = MAX_EXCERPTS + 1  # no-commit plotted as 7


# -------------------------
# Load all run JSON files
# -------------------------


rows: list[dict[str, Any]] = []
turn2_conf_rows: list[dict[str, Any]] = []
turn_conf_rows: list[dict[str, Any]] = []


for path in sorted(RUNS_DIR.glob("*.json")):
   with path.open("r", encoding="utf-8") as f:
       data = json.load(f)


   cond = data.get("condition", {})
   turns = data.get("turns", [])


   rows.append(
       {
           "file": path.name,
           "agent_type": cond.get("agent_type"),
           "reassurance_type": cond.get("reassurance_type"),
           "sequence_file": cond.get("sequence_file"),
           "rep": cond.get("rep"),
           "sequence_name": data.get("sequence_name"),
           "true_author": data.get("true_author"),
           "final_guess": data.get("final_guess"),
           "final_confidence": data.get("final_confidence"),
           "committed": data.get("committed"),
           "toc_turn": data.get("toc_turn"),
           "accuracy": data.get("accuracy"),
           "evidence_requests": data.get("evidence_requests"),
       }
   )


   for turn in turns:
       conf = turn.get("parsed_confidence")
       if conf is not None:
           turn_conf_rows.append(
               {
                   "file": path.name,
                   "agent_type": cond.get("agent_type"),
                   "reassurance_type": cond.get("reassurance_type"),
                   "sequence_name": data.get("sequence_name"),
                   "turn_index": turn.get("turn_index"),
                   "turn_confidence": conf,
               }
           )


   if len(turns) >= 2:
       turn2 = turns[1]
       conf2 = turn2.get("parsed_confidence")
       if conf2 is not None:
           turn2_conf_rows.append(
               {
                   "file": path.name,
                   "agent_type": cond.get("agent_type"),
                   "reassurance_type": cond.get("reassurance_type"),
                   "sequence_name": data.get("sequence_name"),
                   "turn2_confidence": conf2,
               }
           )


df = pd.DataFrame(rows)


if df.empty:
   raise ValueError("No JSON run files found in runs_latest/")


df["toc_plot"] = df["toc_turn"].fillna(NO_COMMIT_PLOT_VALUE)


sequence_order = ["sequence_1", "sequence_2"]


condition_order = [
   ("jtc", "calibrated"),
   ("jtc", "miscalibrated"),
   ("nonjtc", "calibrated"),
   ("nonjtc", "miscalibrated"),
]


condition_labels = [
   "JTC\nCal",
   "JTC\nMiscal",
   "NonJTC\nCal",
   "NonJTC\nMiscal",
]


seq_colors = {
   "sequence_1": "tab:blue",
   "sequence_2": "tab:orange",
}


# -------------------------
# Summary tables
# -------------------------


summary = (
   df.groupby(["agent_type", "reassurance_type", "sequence_name"], dropna=False)
   .agg(
       mean_toc=("toc_plot", "mean"),
       sd_toc=("toc_plot", "std"),
       mean_evidence_requests=("evidence_requests", "mean"),
       sd_evidence_requests=("evidence_requests", "std"),
       mean_accuracy=("accuracy", "mean"),
       sd_accuracy=("accuracy", "std"),
       mean_final_confidence=("final_confidence", "mean"),
       sd_final_confidence=("final_confidence", "std"),
       n=("file", "count"),
   )
   .reset_index()
)


summary.to_csv(FIGURES_DIR / "summary_table.csv", index=False)
print(f"Saved summary table: {FIGURES_DIR / 'summary_table.csv'}")


turn2_df = pd.DataFrame(turn2_conf_rows)
if not turn2_df.empty:
   turn2_summary = (
       turn2_df.groupby(["agent_type", "reassurance_type", "sequence_name"], dropna=False)
       .agg(
           mean_turn2_confidence=("turn2_confidence", "mean"),
           sd_turn2_confidence=("turn2_confidence", "std"),
           n=("file", "count"),
       )
       .reset_index()
   )
   turn2_summary.to_csv(FIGURES_DIR / "turn2_confidence_summary.csv", index=False)
   print(f"Saved turn 2 confidence summary: {FIGURES_DIR / 'turn2_confidence_summary.csv'}")
else:
   turn2_summary = pd.DataFrame()
   print("No turn 2 confidence values found.")


turn_conf_df = pd.DataFrame(turn_conf_rows)
if not turn_conf_df.empty:
   turn_conf_summary = (
       turn_conf_df.groupby(
           ["agent_type", "reassurance_type", "sequence_name", "turn_index"],
           dropna=False
       )
       .agg(
           mean_turn_confidence=("turn_confidence", "mean"),
           sd_turn_confidence=("turn_confidence", "std"),
           n=("file", "count"),
       )
       .reset_index()
   )
   turn_conf_summary.to_csv(FIGURES_DIR / "turn_confidence_trajectory_summary.csv", index=False)
   print(f"Saved confidence trajectory summary: {FIGURES_DIR / 'turn_confidence_trajectory_summary.csv'}")
else:
   turn_conf_summary = pd.DataFrame()
   print("No turn confidence values found.")


# -------------------------
# Load ANOVA results if available
# -------------------------


anova_path = FIGURES_DIR / "anova_results.csv"
if anova_path.exists():
   anova_df = pd.read_csv(anova_path)
   print(f"Loaded ANOVA results from: {anova_path}")
else:
   anova_df = pd.DataFrame()
   print("No anova_results.csv found; significance labels will be omitted.")


# -------------------------
# Helpers
# -------------------------


def mean_sem(values: list[float]):
   arr = np.array(values, dtype=float)
   mean_val = float(np.mean(arr))
   sem_val = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
   return mean_val, sem_val


def values_for_condition(seq_name: str, agent_type: str, reassurance_type: str, metric_col: str):
   sub = df[
       (df["sequence_name"] == seq_name)
       & (df["agent_type"] == agent_type)
       & (df["reassurance_type"] == reassurance_type)
   ]
   return sub[metric_col].dropna().tolist()


def p_to_stars(p: float | None) -> str:
   if p is None or pd.isna(p):
       return ""
   if p < 0.001:
       return "***"
   if p < 0.01:
       return "**"
   if p < 0.05:
       return "*"
   return "ns"


def get_reassurance_star(seq_name: str, measure_name: str) -> str:
   if anova_df.empty:
       return ""
   sub = anova_df[
       (anova_df["sequence_name"] == seq_name)
       & (anova_df["measure"] == measure_name)
       & (anova_df["effect"] == "C(reassurance_type)")
   ]
   if sub.empty:
       return ""
   return p_to_stars(sub.iloc[0]["p_fdr_bh"])


def get_means_for_metric(seq_name: str, metric_col: str):
   means = []
   sems = []
   for agent, reassurance in condition_order:
       vals = values_for_condition(seq_name, agent, reassurance, metric_col)
       m, s = mean_sem(vals)
       means.append(m)
       sems.append(s)
   return np.array(means), np.array(sems)


def add_reassurance_note(ax, measure_name: str):
   s1 = get_reassurance_star("sequence_1", measure_name)
   s2 = get_reassurance_star("sequence_2", measure_name)
   if not s1 and not s2:
       return


   ax.text(
       0.98,
       1.06,
       f"Reassurance effect: seq1 {s1}, seq2 {s2}",
       transform=ax.transAxes,
       ha="right",
       va="bottom",
       fontsize=8.8,
       clip_on=False,
   )


def draw_effect_arrow(ax, x0: float, x1: float, y: float):
   ax.annotate(
       "",
       xy=(x1, y),
       xytext=(x0, y),
       arrowprops=dict(arrowstyle="->", lw=1.2, color="black", alpha=0.85),
   )


def plot_setup_schematic(ax):
   ax.axis("off")
   ax.set_title("Task schematic", fontsize=12, pad=10)


   boxes = [
       (0.08, 0.78, 0.22, 0.10, "Excerpt 1"),
       (0.38, 0.78, 0.22, 0.10, "Excerpt 2"),
       (0.68, 0.78, 0.24, 0.10, "Calibrated vs\nMiscalibrated\nreassurance"),
       (0.38, 0.58, 0.24, 0.10, "Continue\nsampling"),
       (0.68, 0.58, 0.22, 0.10, "Commit"),
   ]


   for x, y, w, h, text in boxes:
       patch = FancyBboxPatch(
           (x, y),
           w,
           h,
           boxstyle="round,pad=0.02",
           edgecolor="black",
           facecolor="white",
           linewidth=1.2,
       )
       ax.add_patch(patch)
       ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.8)


   arrow_kw = dict(arrowstyle="->", lw=1.2, color="black")
   ax.annotate("", xy=(0.38, 0.83), xytext=(0.30, 0.83), arrowprops=arrow_kw)
   ax.annotate("", xy=(0.68, 0.83), xytext=(0.60, 0.83), arrowprops=arrow_kw)
   ax.annotate("", xy=(0.50, 0.68), xytext=(0.78, 0.78), arrowprops=arrow_kw)
   ax.annotate("", xy=(0.78, 0.68), xytext=(0.80, 0.78), arrowprops=arrow_kw)


   ax.text(
       0.08,
       0.38,
       "Agent types\n• JTC\n• NonJTC\n\nMeasures\n• Evidence seeking (ES)\n• Turn of commitment (TOC)\n• Accuracy",
       fontsize=9.2,
       va="top",
   )


def plot_metric_merged(
   ax,
   metric_col: str,
   ylabel: str,
   ylim: tuple[float, float],
   measure_name_for_stars: str,
   add_half_line: bool = False,
   add_no_commit_line: bool = False,
   arrow_direction: str | None = None,
):
   xs = np.arange(len(condition_order))
   offsets = [-0.10, 0.10]


   for offset, seq_name in zip(offsets, sequence_order):
       means, sems = get_means_for_metric(seq_name, metric_col)
       ax.errorbar(
           xs + offset,
           means,
           yerr=sems,
           fmt="o",
           capsize=4,
           markersize=7,
           linewidth=1.8,
           color=seq_colors[seq_name],
           label=seq_name.replace("_", " "),
       )


   ax.set_ylabel(ylabel, fontsize=10)
   ax.set_ylim(*ylim)
   ax.set_xticks(xs)
   ax.set_xticklabels(condition_labels, fontsize=9)


   ax.spines["top"].set_visible(False)
   ax.spines["right"].set_visible(False)
   ax.grid(axis="y", linestyle="--", alpha=0.3)
   ax.axvline(1.5, linestyle="--", alpha=0.3, color="gray")


   if add_half_line:
       ax.axhline(0.5, linewidth=0.8, linestyle="--", alpha=0.45, color="gray")
   if add_no_commit_line:
       ax.axhline(NO_COMMIT_PLOT_VALUE, linewidth=0.8, linestyle="--", alpha=0.45, color="gray")


   add_reassurance_note(ax, measure_name_for_stars)


   if arrow_direction == "down":
       y_arrow = ylim[0] + 0.16 * (ylim[1] - ylim[0])
       draw_effect_arrow(ax, 1.0, 0.0, y_arrow)
       draw_effect_arrow(ax, 3.0, 2.0, y_arrow)
   elif arrow_direction == "up":
       y_arrow = ylim[0] + 0.20 * (ylim[1] - ylim[0])
       draw_effect_arrow(ax, 1.0, 0.0, y_arrow)
       draw_effect_arrow(ax, 3.0, 2.0, y_arrow)


def make_main_story_figure(filename: str):
   fig = plt.figure(figsize=(12.8, 8.6))
   gs = fig.add_gridspec(
       3, 2,
       width_ratios=[1.0, 2.25],
       height_ratios=[1, 1, 1],
       wspace=0.28,
       hspace=0.42,
   )


   ax_schema = fig.add_subplot(gs[:, 0])
   plot_setup_schematic(ax_schema)


   ax_es = fig.add_subplot(gs[0, 1])
   ax_toc = fig.add_subplot(gs[1, 1])
   ax_acc = fig.add_subplot(gs[2, 1])


   plot_metric_merged(
       ax_es,
       metric_col="evidence_requests",
       ylabel="ES",
       ylim=(0, 6.5),
       measure_name_for_stars="evidence_requests",
       arrow_direction="down",
   )
   ax_es.set_title("Main results across both sequences", fontsize=11, pad=12)


   plot_metric_merged(
       ax_toc,
       metric_col="toc_plot",
       ylabel="TOC",
       ylim=(0.5, 7.5),
       measure_name_for_stars="toc_plot",
       add_no_commit_line=True,
       arrow_direction="up",
   )


   plot_metric_merged(
       ax_acc,
       metric_col="accuracy",
       ylabel="Accuracy",
       ylim=(-0.05, 1.05),
       measure_name_for_stars="accuracy",
       add_half_line=True,
       arrow_direction="down",
   )


   handles, labels = ax_es.get_legend_handles_labels()
   ax_es.legend(handles, labels, frameon=False, loc="upper left", fontsize=9)


   fig.text(
       0.60,
       0.015,
       "Points show mean ± SEM. Significance notes indicate FDR-corrected main effects of reassurance within each sequence. "
       "No-commit trials are plotted as 7.",
       ha="center",
       fontsize=9.1,
   )


   out = FIGURES_DIR / filename
   plt.savefig(out, dpi=300, bbox_inches="tight")
   plt.close()
   print(f"Saved plot: {out}")


def plot_confidence_trajectory(filename: str):
   if turn_conf_summary.empty:
       print("Skipping confidence trajectory figure: no confidence trajectory data found.")
       return


   fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)


   line_specs = {
       ("jtc", "calibrated"): ("JTC Cal", "solid"),
       ("jtc", "miscalibrated"): ("JTC Miscal", "dashed"),
       ("nonjtc", "calibrated"): ("NonJTC Cal", "solid"),
       ("nonjtc", "miscalibrated"): ("NonJTC Miscal", "dashed"),
   }


   for ax, seq_name in zip(axes, sequence_order):
       sub_seq = turn_conf_summary[turn_conf_summary["sequence_name"] == seq_name].copy()


       for (agent, reassurance), (label, linestyle) in line_specs.items():
           sub = sub_seq[
               (sub_seq["agent_type"] == agent)
               & (sub_seq["reassurance_type"] == reassurance)
           ].sort_values("turn_index")


           if sub.empty:
               continue


           color = "tab:blue" if agent == "jtc" else "tab:orange"


           ax.errorbar(
               sub["turn_index"],
               sub["mean_turn_confidence"],
               yerr=sub["sd_turn_confidence"] / np.sqrt(sub["n"]),
               fmt="o-",
               linestyle=linestyle,
               color=color,
               capsize=3,
               markersize=4.5,
               linewidth=1.6,
               label=label,
           )


       ax.set_title(seq_name.replace("_", " "), fontsize=10)
       ax.set_xlabel("Turn", fontsize=10)
       ax.set_xticks(range(1, MAX_EXCERPTS + 1))
       ax.set_ylim(40, 90)
       ax.grid(axis="y", linestyle="--", alpha=0.3)
       ax.spines["top"].set_visible(False)
       ax.spines["right"].set_visible(False)
       ax.axvline(2, linestyle="--", alpha=0.3, color="gray")


   axes[0].set_ylabel("Confidence", fontsize=10)
   handles, labels = axes[0].get_legend_handles_labels()
   fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, fontsize=8.5)
   fig.suptitle("Supplementary: confidence trajectory across turns", fontsize=11)


   plt.tight_layout(rect=[0, 0, 1, 0.90])
   out = FIGURES_DIR / filename
   plt.savefig(out, dpi=300, bbox_inches="tight")
   plt.close()
   print(f"Saved plot: {out}")


# -------------------------
# Make figures
# -------------------------


make_main_story_figure("main_story_figure_merged.png")
plot_confidence_trajectory("confidence_trajectory_supplementary.png")
