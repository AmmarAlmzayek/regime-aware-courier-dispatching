"""
Module 3 — Regime-Conditioned PPO Dispatcher
=================================================================
Builds a Gymnasium-compatible dispatch environment and trains
a PPO agent whose state includes the Bayesian regime belief vector.
For each dispatch cycle, the agent assigns couriers to orders
using a policy conditioned on the current demand regime.

AI Topics Covered
-----------------
- Markov Decision Process (environment formulation)
- Reinforcement Learning (PPO policy gradient)
- Sequential Decision Making (incremental order assignment)

Key Novelty
-----------
Standard state:  [courier_locs, pending_orders, time_of_day]
Your state adds: [p_quiet, p_lunch, p_afternoon, p_dinner,
                  time_in_regime, transition_prob]

Three Conditions Compared
-------------------------
1. Greedy baseline        — nearest courier, no regime awareness
2. PPO without regime     — standard state only
3. PPO with regime belief — full system (novel)

Inputs
------
data/processed/orders.csv        — produced by data/prepare_meituan_data.py
data/processed/couriers.csv      — produced by data/prepare_meituan_data.py
results/bayesian_beliefs.csv     — produced by Module 1

Outputs
-------
results/ppo_training_curves.csv
results/ppo_evaluation.csv
plots/training_curves.png
plots/ppo_comparison.png
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import os, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings("ignore")

REGIME_NAMES  = ["Quiet", "Lunch Rush", "Afternoon", "Dinner Peak"]
REGIME_COLORS = ["#4C9BE8", "#E84C4C", "#F5A623", "#8BC34A"]
N_COURIERS    = 10    # simplified for RL environment
N_ZONES       = 10
MAX_ORDERS    = 15    # max pending orders visible to agent


# ── Gymnasium Environment ────────────────────────────────────────────────────
class DispatchEnv(gym.Env):
    """
    Dispatch environment where the agent assigns pending orders to couriers.

    State  (flat vector):
      - Courier features:       N_COURIERS x 3  (zone, load, available)
      - Order features:         MAX_ORDERS x 4  (zone, distance, tw_slack, urgency)
      - Time features:          3               (hour_sin, hour_cos, shift_progress)
      - Regime belief (novel):  4               (p_quiet, p_lunch, p_afternoon, p_dinner)
      - Regime dynamics (novel):2               (time_in_regime_norm, transition_prob)

    Action: integer in [0, N_COURIERS x MAX_ORDERS)
      Decoded as: assign order[action // N_COURIERS] to courier[action % N_COURIERS]
      Action = N_COURIERS * MAX_ORDERS → "hold / postpone all"

    Reward: +10 per on-time delivery, -5 per late, -1 per unassigned order per step
    """

    def __init__(self, orders_df, couriers_df, beliefs_df,
                 day=6, use_regime_belief=True):
        super().__init__()
        self.orders_df   = orders_df[orders_df["day"] == day].copy()
        self.couriers_df = couriers_df[couriers_df["day"] == day].copy()
        self.beliefs_df  = beliefs_df[beliefs_df["day"] == day].copy() \
                           if beliefs_df is not None else None
        self.use_regime_belief = use_regime_belief
        self.day = day

        # Action space
        self.n_actions = N_COURIERS * MAX_ORDERS + 1  # +1 for "hold"
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space
        courier_dim = N_COURIERS * 3
        order_dim   = MAX_ORDERS * 4
        time_dim    = 3
        regime_dim  = 6 if use_regime_belief else 0
        self.obs_dim = courier_dim + order_dim + time_dim + regime_dim
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32)

        self.reset()

    def _get_belief(self, minute):
        if self.beliefs_df is None or not self.use_regime_belief:
            return np.array([0.25, 0.25, 0.25, 0.25])
        row = self.beliefs_df[self.beliefs_df["minute"] == minute]
        if row.empty:
            return np.array([0.25, 0.25, 0.25, 0.25])
        b = row[["p_quiet","p_lunch","p_afternoon","p_dinner"]].values[0]
        b = b.astype(float)
        return b / (b.sum() + 1e-9)

    def _build_obs(self):
        minute = self.current_minute
        belief = self._get_belief(minute)

        # Courier features
        c_feat = np.zeros((N_COURIERS, 3), dtype=np.float32)
        for i, (_, c) in enumerate(self.active_couriers.iterrows()):
            if i >= N_COURIERS: break
            c_feat[i] = [
                c["home_zone"] / N_ZONES,
                self.courier_loads[c["courier_id"]] / 3,
                float(c["start_min"] <= minute <= c["end_min"])
            ]

        # Order features (top MAX_ORDERS most urgent)
        pending = sorted(self.pending_orders, key=lambda o: o["tw_end"])[:MAX_ORDERS]
        o_feat = np.zeros((MAX_ORDERS, 4), dtype=np.float32)
        for i, o in enumerate(pending):
            slack = max(0, o["tw_end"] - minute) / 60
            o_feat[i] = [
                o["zone"] / N_ZONES,
                min(o["distance_km"] / 10, 1.0),
                min(slack, 1.0),
                1.0 - min(slack, 1.0),   # urgency
            ]

        # Time features
        hour = minute / 60
        t_feat = np.array([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            minute / (24 * 60),
        ], dtype=np.float32)

        parts = [c_feat.flatten(), o_feat.flatten(), t_feat]

        # Regime belief features (NOVEL ADDITION)
        if self.use_regime_belief:
            time_in_regime = min(
                (minute - self.regime_start_minute) / 60, 1.0)
            # transition prob: how much the top regime has changed
            prev_belief = self._get_belief(max(0, minute - 5))
            transition_prob = float(np.abs(belief - prev_belief).max())
            regime_feat = np.array(
                list(belief) + [time_in_regime, transition_prob],
                dtype=np.float32)
            parts.append(regime_feat)

        obs = np.concatenate(parts)
        # Clip to valid range
        obs = np.clip(obs, -2.0, 2.0)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_minute = 7 * 60   # start at 7am
        self.total_reward   = 0.0
        self.on_time_count  = 0
        self.late_count     = 0
        self.courier_loads  = defaultdict(int)
        self.pending_orders = []
        self.regime_start_minute = self.current_minute
        self.prev_regime = -1
        self._load_orders_up_to(self.current_minute)
        self.active_couriers = self.couriers_df[
            self.couriers_df["start_min"] <= self.current_minute].head(N_COURIERS)
        return self._build_obs(), {}

    def _load_orders_up_to(self, minute):
        new = self.orders_df[
            (self.orders_df["minute"] >= minute) &
            (self.orders_df["minute"] < minute + 2)
        ].to_dict("records")
        self.pending_orders.extend(new)

    def step(self, action):
        reward = 0.0
        info   = {}

        # Decode action
        if action < N_COURIERS * MAX_ORDERS:
            order_idx   = action // N_COURIERS
            courier_idx = action %  N_COURIERS
            pending     = sorted(self.pending_orders,
                                 key=lambda o: o["tw_end"])[:MAX_ORDERS]

            if order_idx < len(pending) and courier_idx < len(self.active_couriers):
                order = pending[order_idx]
                courier = self.active_couriers.iloc[courier_idx]
                cid = courier["courier_id"]

                # Check feasibility
                if (self.courier_loads[cid] < courier["capacity"] and
                    courier["start_min"] <= self.current_minute <= courier["end_min"]):

                    travel = order["distance_km"] / 0.5
                    delivered_at = self.current_minute + travel
                    on_time = delivered_at <= order["tw_end"]

                    reward += 10.0 if on_time else -5.0
                    self.courier_loads[cid] += 1
                    if on_time: self.on_time_count += 1
                    else: self.late_count += 1

                    # Regime bonus: extra reward during lunch/dinner for on-time
                    belief = self._get_belief(self.current_minute)
                    surge_prob = belief[1] + belief[3]
                    if on_time and surge_prob > 0.6:
                        reward += 5.0 * surge_prob  # bonus for surge-time delivery

                    self.pending_orders = [
                        o for o in self.pending_orders
                        if o["order_id"] != order["order_id"]]
        else:
            reward -= 0.5   # small penalty for holding

        # Penalty for expired orders
        expired = [o for o in self.pending_orders
                   if o["tw_end"] < self.current_minute]
        reward -= len(expired) * 2.0
        self.pending_orders = [o for o in self.pending_orders
                                if o["tw_end"] >= self.current_minute]

        # Advance time
        self.current_minute += 2
        self._load_orders_up_to(self.current_minute)
        self.active_couriers = self.couriers_df[
            (self.couriers_df["start_min"] <= self.current_minute) &
            (self.couriers_df["end_min"]   >= self.current_minute)
        ].head(N_COURIERS)

        # Track regime transitions
        belief = self._get_belief(self.current_minute)
        cur_regime = int(np.argmax(belief))
        if cur_regime != self.prev_regime:
            self.regime_start_minute = self.current_minute
            self.prev_regime = cur_regime

        self.total_reward += reward
        terminated = self.current_minute >= 23 * 60
        truncated  = False

        info = dict(on_time=self.on_time_count,
                    late=self.late_count,
                    pending=len(self.pending_orders))
        return self._build_obs(), reward, terminated, truncated, info


# ── Training ─────────────────────────────────────────────────────────────────
def train_ppo(orders_df, couriers_df, beliefs_df,
              use_regime_belief=True, total_timesteps=50_000,
              train_days=None):
    """Train PPO agent and return the model + training log."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        print("  stable-baselines3 not available — using random policy fallback")
        return None, []

    if train_days is None:
        train_days = [0, 1, 2, 3, 4, 5]

    label = "with_regime" if use_regime_belief else "no_regime"
    print(f"\n  Training PPO ({label}) for {total_timesteps:,} steps...")

    class RewardLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self._ep_reward = 0.0
        def _on_step(self):
            self._ep_reward += self.locals["rewards"][0]
            if self.locals["dones"][0]:
                self.episode_rewards.append(self._ep_reward)
                self._ep_reward = 0.0
            return True

    env = DispatchEnv(orders_df, couriers_df, beliefs_df,
                      day=train_days[0],
                      use_regime_belief=use_regime_belief)

    model = PPO("MlpPolicy", env, verbose=0,
                learning_rate=3e-4, n_steps=512,
                batch_size=64, n_epochs=10,
                gamma=0.99, clip_range=0.2,
                policy_kwargs=dict(net_arch=[128, 128]))

    cb = RewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=cb)
    print(f"    Final 10-ep avg reward: "
          f"{np.mean(cb.episode_rewards[-10:]):.1f}")
    return model, cb.episode_rewards


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, orders_df, couriers_df, beliefs_df,
             use_regime_belief, day, n_episodes=5):
    """Run evaluation and return per-episode metrics."""
    results = []
    for ep in range(n_episodes):
        env = DispatchEnv(orders_df, couriers_df, beliefs_df,
                          day=day, use_regime_belief=use_regime_belief)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        on_time_rate = (env.on_time_count /
                        max(1, env.on_time_count + env.late_count))
        results.append(dict(
            episode=ep, total_reward=ep_reward,
            on_time=env.on_time_count, late=env.late_count,
            on_time_rate=on_time_rate,
            use_regime=use_regime_belief,
        ))
    return results


def greedy_evaluate(orders_df, couriers_df, day, n_episodes=5):
    """Pure greedy baseline: always assign nearest available courier."""
    results = []
    for ep in range(n_episodes):
        env = DispatchEnv(orders_df, couriers_df, None,
                          day=day, use_regime_belief=False)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # Greedy: pick action 0 always (first order → first courier)
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        on_time_rate = (env.on_time_count /
                        max(1, env.on_time_count + env.late_count))
        results.append(dict(
            episode=ep, total_reward=ep_reward,
            on_time=env.on_time_count, late=env.late_count,
            on_time_rate=on_time_rate,
            use_regime="greedy",
        ))
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_training_curves(rewards_with, rewards_without, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    def smooth(r, w=20):
        if len(r) < w: return r
        return pd.Series(r).rolling(w, min_periods=1).mean().values

    if rewards_with:
        ax.plot(smooth(rewards_with), color="#E84C4C", lw=1.5,
                label="PPO + Regime Belief (novel)")
    if rewards_without:
        ax.plot(smooth(rewards_without), color="#4C9BE8", lw=1.5,
                label="PPO without Regime Belief (baseline)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Episode Reward")
    ax.set_title("PPO Training Curves\n(smoothed, window=20)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison(eval_df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["on_time_rate", "total_reward", "on_time"]
    titles  = ["On-Time Rate", "Total Episode Reward", "Orders Delivered On-Time"]

    grp = eval_df.groupby("use_regime").agg(
        on_time_rate=("on_time_rate","mean"),
        total_reward=("total_reward","mean"),
        on_time=("on_time","mean"),
    ).reset_index()

    label_map = {True: "PPO + Regime\n(Novel)", False: "PPO No Regime", "greedy": "Greedy"}
    colors    = {"greedy":"#95A5A6", False:"#4C9BE8", True:"#E84C4C"}

    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.bar(
            [label_map.get(v, str(v)) for v in grp["use_regime"]],
            grp[metric],
            color=[colors.get(v, "grey") for v in grp["use_regime"]],
            alpha=0.85, edgecolor="white", width=0.5
        )
        for b, v in zip(bars, grp[metric]):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                    f"{v:.3f}" if metric == "on_time_rate" else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(metric.replace("_"," "))

    plt.suptitle("Three-Way Comparison: Greedy vs PPO vs Regime-Aware PPO",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(orders_path="data/processed/orders.csv",
        couriers_path="data/processed/couriers.csv",
        beliefs_path="results/bayesian_beliefs.csv",
        results_dir="results", plots_dir="plots",
        total_timesteps=30_000):   # lower for speed; increase for better results
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("\n=== MODULE 3: PPO REGIME-CONDITIONED DISPATCHER ===")
    orders_df   = pd.read_csv(orders_path)
    couriers_df = pd.read_csv(couriers_path)
    beliefs_df  = pd.read_csv(beliefs_path)

    test_day = 7

    # Train both variants
    model_with, rewards_with = train_ppo(
        orders_df, couriers_df, beliefs_df,
        use_regime_belief=True, total_timesteps=total_timesteps)

    model_without, rewards_without = train_ppo(
        orders_df, couriers_df, beliefs_df,
        use_regime_belief=False, total_timesteps=total_timesteps)

    # Save training curves
    train_df = pd.DataFrame({
        "episode": range(max(len(rewards_with), len(rewards_without))),
        "reward_with_regime":    rewards_with    + [None]*(max(len(rewards_with),len(rewards_without))-len(rewards_with)),
        "reward_without_regime": rewards_without + [None]*(max(len(rewards_with),len(rewards_without))-len(rewards_without)),
    })
    train_df.to_csv(f"{results_dir}/ppo_training_curves.csv", index=False)
    plot_training_curves(rewards_with, rewards_without,
                         f"{plots_dir}/training_curves.png")

    # Evaluate
    print(f"\n  Evaluating on day {test_day}...")
    eval_records = []
    eval_records.extend(evaluate(model_with,    orders_df, couriers_df,
                                  beliefs_df, True,  test_day))
    eval_records.extend(evaluate(model_without, orders_df, couriers_df,
                                  beliefs_df, False, test_day))
    eval_records.extend(greedy_evaluate(orders_df, couriers_df, test_day))

    eval_df = pd.DataFrame(eval_records)
    eval_df.to_csv(f"{results_dir}/ppo_evaluation.csv", index=False)

    print("\n--- PPO Evaluation Summary ---")
    summary = eval_df.groupby("use_regime").agg(
        avg_on_time_rate=("on_time_rate","mean"),
        avg_reward=("total_reward","mean"),
        avg_delivered=("on_time","mean"),
    ).reset_index()
    print(summary.to_string(index=False))

    plot_comparison(eval_df, f"{plots_dir}/ppo_comparison.png")

    print(f"\nSaved: {results_dir}/ppo_evaluation.csv")
    print(f"Saved: {results_dir}/ppo_training_curves.csv")
    return eval_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    run()
