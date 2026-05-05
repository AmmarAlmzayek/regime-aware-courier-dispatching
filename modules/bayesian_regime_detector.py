"""
Module 1 — Bayesian Regime Detector
=================================================================
Uses a Bayesian forward filter with hand-tuned emission parameters
fitted to real Meituan order patterns. For each minute of the test
days, the filter maintains a probability distribution over the 4
hidden demand regimes and updates it as new observations arrive.

AI Topics Covered
-----------------
- Bayesian belief update (forward algorithm)
- Probabilistic Reasoning over Time
- Probabilistic inference under uncertainty

Inputs
------
data/processed/observations.csv  — produced by data/prepare_meituan_data.py

Outputs
-------
results/bayesian_beliefs.csv          — posterior P(regime | history) per timestep
results/bayesian_metrics.csv          — accuracy and confidence stats
plots/bayesian_beliefs_day{N}.png  — belief plot for first test day
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import classification_report
import os
import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

REGIME_NAMES  = ["Quiet", "Lunch Rush", "Afternoon", "Dinner Peak"]
REGIME_COLORS = ["#4C9BE8", "#E84C4C", "#F5A623", "#8BC34A"]

# ── Emission parameters ───────────────────────────────────────────────────────
# Hand-tuned to match real Meituan order patterns:
# orders_mu/std: expected normalized order count per minute per regime
# hour_mu/std:   expected hour of day when this regime is active

REGIME_PARAMS = {
    0: dict(orders_mu=0.19, orders_std=0.15, hour_mu=4.0,  hour_std=4.0),   # Quiet
    1: dict(orders_mu=0.75, orders_std=0.20, hour_mu=11.5, hour_std=1.0),   # Lunch Rush
    2: dict(orders_mu=0.42, orders_std=0.15, hour_mu=15.0, hour_std=1.0),   # Afternoon
    3: dict(orders_mu=0.66, orders_std=0.20, hour_mu=19.0, hour_std=2.0),   # Dinner Peak
}

# ── Transition matrix ─────────────────────────────────────────────────────────
# How likely is it to stay in the same regime vs switch to another
# Rows = current regime, Columns = next regime

TRANSMAT = np.array([
    [0.92, 0.03, 0.02, 0.03],   # Quiet      → mostly stays quiet
    [0.05, 0.88, 0.05, 0.02],   # Lunch Rush → mostly stays lunch rush
    [0.05, 0.05, 0.85, 0.05],   # Afternoon  → mostly stays afternoon
    [0.03, 0.02, 0.05, 0.90],   # Dinner     → mostly stays dinner peak
])

# ── Step 1: Emission probability ──────────────────────────────────────────────

def compute_emission(row):
    """
    For a given observation (one minute), compute how likely
    it is under each of the 4 regimes.
    Uses order count and time of day as the two signals.
    """
    n    = row["orders_per_min"]
    hour = row["minute"] / 60.0
    probs = np.zeros(4)
    for k, p in REGIME_PARAMS.items():
        p_orders = norm.pdf(n,    p["orders_mu"], p["orders_std"] + 1e-6)
        p_hour   = norm.pdf(hour, p["hour_mu"],   p["hour_std"]  + 1e-6)
        probs[k] = p_orders * p_hour + 1e-10
    return probs / probs.sum()


# ── Step 2: Bayesian forward filter ───────────────────────────────────────────

def bayesian_forward(day_df):
    """
    Run the Bayesian forward algorithm minute by minute.

    At each minute:
      1. Predict next belief using transition matrix
      2. Update using emission probability of current observation
      3. Smooth slightly to avoid overconfidence

    Returns beliefs array of shape (T, 4).
    """
    T = len(day_df)
    beliefs = np.zeros((T, 4))

    # Set initial belief based on time of day
    first_hour = day_df.iloc[0]["minute"] / 60
    if first_hour < 10:
        belief = np.array([0.70, 0.10, 0.10, 0.10])   # probably quiet
    elif first_hour < 14:
        belief = np.array([0.10, 0.60, 0.20, 0.10])   # probably lunch rush
    elif first_hour < 17:
        belief = np.array([0.20, 0.10, 0.60, 0.10])   # probably afternoon
    else:
        belief = np.array([0.10, 0.10, 0.10, 0.70])   # probably dinner peak

    for t, (_, row) in enumerate(day_df.iterrows()):
        # Predict: where might we be given where we were
        predicted = TRANSMAT.T @ belief

        # Update: weight by how likely this observation is per regime
        emission = compute_emission(row)
        updated  = predicted * emission
        total    = updated.sum()
        belief   = updated / total if total > 0 else predicted

        # Smooth: prevent belief from collapsing to 100% one regime
        belief = 0.85 * belief + 0.05 * np.ones(4)
        belief = belief / belief.sum()

        beliefs[t] = belief

    return beliefs


# ── Step 3: Plotting ──────────────────────────────────────────────────────────

def plot_beliefs(beliefs, true_regimes, day, save_path):
    """Plot stacked belief area + predicted vs true for one day."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    T     = len(beliefs)
    hours = np.arange(T) / 60

    # Panel 1 — stacked belief area
    ax = axes[0]
    bottom = np.zeros(T)
    for k in range(4):
        ax.fill_between(hours, bottom, bottom + beliefs[:, k],
                        alpha=0.8, color=REGIME_COLORS[k], label=REGIME_NAMES[k])
        bottom += beliefs[:, k]
    ax.set_ylabel("P(regime | history)")
    ax.set_title(f"Regime Beliefs — Day {day} (Bayesian Forward Filter)")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.set_ylim(0, 1)

    # Panel 2 — predicted vs true
    ax = axes[1]
    pred = np.argmax(beliefs, axis=1)
    ax.step(hours, pred,         where="post", color="navy",   lw=1.5, label="Predicted")
    ax.step(hours, true_regimes, where="post", color="crimson",
            lw=1.2, alpha=0.6, linestyle="--", label="True")
    ax.set_yticks(range(4))
    ax.set_yticklabels(REGIME_NAMES, fontsize=8)
    ax.set_ylabel("Regime")
    ax.set_xlabel("Hour of day")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(obs_path="data/processed/observations.csv",
        results_dir="results",
        plots_dir="plots"):

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir,   exist_ok=True)

    print("\n=== MODULE 1: BAYESIAN REGIME DETECTOR ===")

    if not os.path.exists(obs_path):
        raise FileNotFoundError(
            f"{obs_path} not found. Run prepare_meituan_data.py first."
        )
    obs_df = pd.read_csv(obs_path)
    print(f"  Loaded observations.csv: {len(obs_df):,} rows")

    # Split days — train / test
    days       = sorted(obs_df["day"].unique())
    train_days = days[:6]
    test_days  = days[6:]
    print(f"  Train days: {train_days}")
    print(f"  Test days:  {test_days}")

    # Run Bayesian forward filter on test days
    all_beliefs, all_true, all_pred = [], [], []
    belief_rows = []

    print("\n  Running Bayesian forward filter on test days...")
    for day in test_days:
        sub  = obs_df[obs_df["day"] == day].sort_values("minute").reset_index(drop=True)
        true = sub["regime"].values

        beliefs = bayesian_forward(sub)
        pred    = np.argmax(beliefs, axis=1)

        all_beliefs.append(beliefs)
        all_true.extend(true.tolist())
        all_pred.extend(pred.tolist())

        for t, (b, tr) in enumerate(zip(beliefs, true)):
            belief_rows.append(dict(
                day=day,
                minute=t,
                p_quiet=round(b[0], 4),
                p_lunch=round(b[1], 4),
                p_afternoon=round(b[2], 4),
                p_dinner=round(b[3], 4),
                predicted_regime=int(np.argmax(b)),
                true_regime=int(tr),
            ))

        # Plot first test day
        if day == test_days[0]:
            plot_beliefs(beliefs, true, day,
                         f"{plots_dir}/bayesian_beliefs_day{day}.png")

    # Accuracy
    acc = np.mean(np.array(all_true) == np.array(all_pred))
    print(f"\n--- Results ---")
    print(f"  Overall accuracy: {acc:.1%}")
    print()
    print(classification_report(
        all_true, all_pred,
        target_names=REGIME_NAMES, digits=3
    ))

    # Confidence stats
    beliefs_df = pd.DataFrame(belief_rows)
    conf_cols  = ["p_quiet", "p_lunch", "p_afternoon", "p_dinner"]
    wrong = beliefs_df[beliefs_df["predicted_regime"] != beliefs_df["true_regime"]]
    right = beliefs_df[beliefs_df["predicted_regime"] == beliefs_df["true_regime"]]
    wc = wrong[conf_cols].max(axis=1)
    rc = right[conf_cols].max(axis=1)
    print(f"  When WRONG:   mean confidence = {wc.mean():.3f}")
    print(f"  When CORRECT: mean confidence = {rc.mean():.3f}")

    # Save outputs
    beliefs_df.to_csv(f"{results_dir}/bayesian_beliefs.csv", index=False)
    print(f"\n  Saved: {results_dir}/bayesian_beliefs.csv")

    metrics = dict(
        accuracy=round(float(acc), 4),
        mean_confidence_correct=round(float(rc.mean()), 4),
        mean_confidence_wrong=round(float(wc.mean()), 4),
    )
    pd.DataFrame([metrics]).to_csv(f"{results_dir}/bayesian_metrics.csv", index=False)
    print(f"  Saved: {results_dir}/bayesian_metrics.csv")

    return beliefs_df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run()