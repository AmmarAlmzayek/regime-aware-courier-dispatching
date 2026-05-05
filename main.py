"""
MAIN PIPELINE — Regime-Aware Courier Dispatching
=================================================
Runs all 4 modules in sequence:
  Step 0: Generate data
  Step 1: HMM regime detection      (Person 1)
  Step 2: CSP + A* assignment       (Person 2)
  Step 3: PPO regime-conditioned RL (Person 3)
  Step 4: Final results aggregation

Usage:
  python main.py              # full run
  python main.py --fast       # reduced timesteps for quick test
  python main.py --step 1     # run only step 1
"""

import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
PLOTS_DIR   = os.path.join(BASE, "plots")

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.chdir(BASE)


def step0_generate_data():
    print("\n" + "="*55)
    print("  STEP 0: Generating synthetic dispatch data")
    print("="*55)
    sys.path.insert(0, DATA_DIR)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_data", os.path.join(DATA_DIR, "generate_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__name__ = "generate_data"

    import numpy as np
    import pandas as pd
    from scipy.stats import entropy as scipy_entropy

    np.random.seed(42)
    all_orders, all_couriers = [], []
    for day in range(mod.N_DAYS):
        all_orders.extend(mod.generate_orders(day))
        all_couriers.extend(mod.generate_couriers(day))

    orders_df   = pd.DataFrame(all_orders)
    couriers_df = pd.DataFrame(all_couriers)
    obs_df      = mod.build_hmm_observations(orders_df)

    orders_df.to_csv(  os.path.join(DATA_DIR, "orders.csv"),       index=False)
    couriers_df.to_csv(os.path.join(DATA_DIR, "couriers.csv"),     index=False)
    obs_df.to_csv(     os.path.join(DATA_DIR, "observations.csv"), index=False)

    print(f"  ✓ {len(orders_df):,} orders, {len(couriers_df)} courier records")
    print(f"  ✓ Regime distribution:\n"
          f"    {dict(orders_df['regime'].value_counts())}")


def step1_hmm():
    print("\n" + "="*55)
    print("  STEP 1: HMM Regime Detector (Person 1)")
    print("="*55)
    sys.path.insert(0, os.path.join(BASE, "modules"))
    import hmm_regime_detector as m1
    m1.run(obs_path   =os.path.join(DATA_DIR, "observations.csv"),
           results_dir=RESULTS_DIR,
           plots_dir  =PLOTS_DIR)


def step2_csp():
    print("\n" + "="*55)
    print("  STEP 2: CSP + A* Assignment Engine (Person 2)")
    print("="*55)
    import csp_assignment as m2
    m2.run(orders_path  =os.path.join(DATA_DIR, "orders.csv"),
           couriers_path=os.path.join(DATA_DIR, "couriers.csv"),
           beliefs_path =os.path.join(RESULTS_DIR, "hmm_beliefs.csv"),
           results_dir  =RESULTS_DIR,
           plots_dir    =PLOTS_DIR)


def step3_ppo(fast=False):
    print("\n" + "="*55)
    print("  STEP 3: PPO Regime-Conditioned Dispatcher (Person 3)")
    print("="*55)
    import ppo_dispatcher as m3
    timesteps = 15_000 if fast else 40_000
    m3.run(orders_path  =os.path.join(DATA_DIR, "orders.csv"),
           couriers_path=os.path.join(DATA_DIR, "couriers.csv"),
           beliefs_path =os.path.join(RESULTS_DIR, "hmm_beliefs.csv"),
           results_dir  =RESULTS_DIR,
           plots_dir    =PLOTS_DIR,
           total_timesteps=timesteps)


def step4_aggregate():
    print("\n" + "="*55)
    print("  STEP 4: Final Results Aggregation")
    print("="*55)
    import results_aggregator as m4
    m4.run(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║  REGIME-AWARE COURIER DISPATCHING                    ║
║  HMM + Constrained A* + Regime-Conditioned PPO       ║
║                                                      ║
║  Novel: HMM belief fed into RL state space           ║
║  Dataset: Synthetic Meituan-style (8 days, 654K+)    ║
╚══════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Reduce PPO timesteps for quick test run")
    parser.add_argument("--step", type=int, default=0,
                        help="Run only a specific step (1-4). 0=all")
    args = parser.parse_args()

    print_banner()
    t_start = time.time()

    if args.step == 0 or args.step == 0:
        step0_generate_data()
    if args.step == 0 or args.step == 1:
        step1_hmm()
    if args.step == 0 or args.step == 2:
        step2_csp()
    if args.step == 0 or args.step == 3:
        step3_ppo(fast=args.fast)
    if args.step == 0 or args.step == 4:
        step4_aggregate()

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Results → {RESULTS_DIR}/")
    print(f"  Plots   → {PLOTS_DIR}/")
    print(f"{'='*55}")
