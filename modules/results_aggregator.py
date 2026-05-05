"""
Module 4 — Final Results Aggregator
=====================================
Combines outputs from all three modules into a single
summary report with the key paper-ready figures.

Outputs:
  plots/final_summary.png   — the main result figure
  results/final_summary.csv — all numbers in one table
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

REGIME_NAMES  = ["Quiet", "Lunch Rush", "Afternoon", "Dinner Peak"]
REGIME_COLORS = ["#4C9BE8", "#E84C4C", "#F5A623", "#8BC34A"]


def run(results_dir="results", plots_dir="plots"):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("\n=== MODULE 4: FINAL RESULTS AGGREGATOR ===")

    # Load all results
    hmm_metrics  = pd.read_csv(f"{results_dir}/hmm_metrics.csv")
    csp_metrics  = pd.read_csv(f"{results_dir}/csp_metrics.csv")
    ppo_eval     = pd.read_csv(f"{results_dir}/ppo_evaluation.csv")
    hmm_beliefs  = pd.read_csv(f"{results_dir}/hmm_beliefs.csv")

    # ── Figure: 2×2 summary ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    # ── Panel A: HMM regime belief (one day, 6am–11pm) ─────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    day_data = hmm_beliefs[hmm_beliefs["day"] == hmm_beliefs["day"].max()].copy()
    day_data = day_data[(day_data["minute"] >= 360) & (day_data["minute"] <= 1380)]
    hours = day_data["minute"] / 60
    cols  = ["p_quiet","p_lunch","p_afternoon","p_dinner"]
    bottom = np.zeros(len(day_data))
    for k, (col, name, color) in enumerate(zip(cols, REGIME_NAMES, REGIME_COLORS)):
        vals = day_data[col].values
        ax_a.fill_between(hours, bottom, bottom + vals,
                          alpha=0.82, color=color, label=name)
        bottom += vals
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Hour of day")
    ax_a.set_ylabel("P(regime | observations)")
    ax_a.set_title("A — HMM Online Regime Beliefs\n(Bayesian forward algorithm)")
    ax_a.legend(fontsize=7, loc="upper left", ncol=2)
    acc = hmm_metrics["accuracy"].values[0]
    ax_a.text(0.98, 0.04, f"Accuracy: {acc:.1%}",
              transform=ax_a.transAxes, ha="right", fontsize=9,
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # ── Panel B: A* vs Greedy on-time rate per regime ───────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    csp_astar  = csp_metrics[csp_metrics["method"] == "astar"]
    csp_greedy = csp_metrics[csp_metrics["method"] == "greedy"]
    x = np.arange(len(REGIME_NAMES))
    w = 0.35
    bars1 = ax_b.bar(x - w/2,
                     [csp_greedy[csp_greedy["regime"]==r]["on_time_rate"].values[0]
                      if r in csp_greedy["regime"].values else 0 for r in range(4)],
                     w, label="Greedy", color="#95A5A6", alpha=0.85)
    bars2 = ax_b.bar(x + w/2,
                     [csp_astar[csp_astar["regime"]==r]["on_time_rate"].values[0]
                      if r in csp_astar["regime"].values else 0 for r in range(4)],
                     w, label="A* + Regime Weights", color="#E84C4C", alpha=0.85)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(REGIME_NAMES, rotation=15, fontsize=8)
    ax_b.set_ylabel("On-Time Rate")
    ax_b.set_ylim(0, 1.15)
    ax_b.set_title("B — CSP A* vs Greedy\nOn-Time Rate by Demand Regime")
    ax_b.legend(fontsize=8)
    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            ax_b.text(b.get_x() + b.get_width()/2, h + 0.01,
                      f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    # ── Panel C: PPO training curves ─────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    try:
        curves = pd.read_csv(f"{results_dir}/ppo_training_curves.csv")
        def smooth(s, w=15):
            return s.rolling(w, min_periods=1).mean()
        if "reward_with_regime" in curves.columns:
            with_r = curves["reward_with_regime"].dropna()
            without_r = curves["reward_without_regime"].dropna()
            ax_c.plot(smooth(with_r),    color="#E84C4C", lw=1.5,
                      label="PPO + Regime Belief")
            ax_c.plot(smooth(without_r), color="#4C9BE8", lw=1.5,
                      label="PPO (No Regime)")
    except Exception:
        ax_c.text(0.5, 0.5, "Training curves\nnot available",
                  transform=ax_c.transAxes, ha="center")
    ax_c.set_xlabel("Episode")
    ax_c.set_ylabel("Total Episode Reward")
    ax_c.set_title("C — PPO Training Curves\n(smoothed, window=15)")
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.25)

    # ── Panel D: Three-way final comparison ──────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ppo_summary = ppo_eval.groupby("use_regime")["on_time_rate"].mean()
    labels, vals, bar_colors = [], [], []
    label_map = {"greedy":"Greedy\n(baseline)",
                 "False":"PPO\n(no regime)",
                 "True":"PPO + Regime\nBelief (ours)"}
    color_map  = {"greedy":"#95A5A6", "False":"#4C9BE8", "True":"#E84C4C"}
    for k, v in ppo_summary.items():
        labels.append(label_map.get(str(k), str(k)))
        vals.append(v)
        bar_colors.append(color_map.get(str(k), "grey"))
    bars = ax_d.bar(labels, vals, color=bar_colors, alpha=0.88,
                    edgecolor="white", width=0.45)
    for b, v in zip(bars, vals):
        ax_d.text(b.get_x() + b.get_width()/2, v + 0.005,
                  f"{v:.1%}", ha="center", va="bottom",
                  fontsize=10, fontweight="bold")
    ax_d.set_ylim(0, 1.15)
    ax_d.set_ylabel("Mean On-Time Rate")
    ax_d.set_title("D — Final Three-Way Comparison\nMean On-Time Rate (test day)")

    # Novelty annotation
    if len(vals) >= 2:
        best  = max(vals)
        second= sorted(vals)[-2] if len(vals) > 1 else best
        delta = best - second
        ax_d.annotate(f"+{delta:.1%}\nvs best baseline",
                      xy=(labels[vals.index(best)], best),
                      xytext=(0.72, 0.6), textcoords="axes fraction",
                      arrowprops=dict(arrowstyle="->", color="black"),
                      fontsize=9, color="darkred", fontweight="bold")

    plt.suptitle(
        "Regime-Aware Courier Dispatching via HMM + Constrained A* + PPO\n"
        "Novel: HMM regime belief fed directly into RL state space",
        fontsize=12, y=1.01, fontweight="bold")

    plt.savefig(f"{plots_dir}/final_summary.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plots_dir}/final_summary.png")

    # ── Save consolidated CSV ─────────────────────────────────────────────────
    rows = []
    rows.append(dict(module="HMM", metric="regime_accuracy",
                     value=round(float(hmm_metrics["accuracy"].values[0]), 4)))
    if "mean_detection_lag" in hmm_metrics.columns:
        lag = hmm_metrics["mean_detection_lag"].values[0]
        if pd.notna(lag):
            rows.append(dict(module="HMM", metric="mean_detection_lag_min",
                             value=round(float(lag), 2)))

    for _, row in csp_metrics.iterrows():
        rows.append(dict(
            module=f"CSP_{row['method']}",
            metric=f"on_time_rate_regime{row['regime']}",
            value=round(float(row["on_time_rate"]), 4)
        ))

    for k, v in ppo_summary.items():
        rows.append(dict(module="PPO", metric=f"on_time_rate_{k}",
                         value=round(float(v), 4)))

    final_df = pd.DataFrame(rows)
    final_df.to_csv(f"{results_dir}/final_summary.csv", index=False)
    print(f"  Saved: {results_dir}/final_summary.csv")

    print("\n=== ALL RESULTS SUMMARY ===")
    print(final_df.to_string(index=False))
    return final_df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    run()
