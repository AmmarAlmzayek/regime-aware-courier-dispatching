"""
MAIN PIPELINE — Regime-Aware Courier Dispatching
=================================================
Runs all modules in sequence on real Meituan INFORMS TSL data.

  Step 0: Prepare real Meituan data    (data/prepare_meituan_data.py)
  Step 1: HMM regime detection         (modules/hmm_regime_detector.py)
  Step 2: CSP + A* assignment          (modules/csp_assignment.py)
  Step 3: PPO regime-conditioned RL    (modules/ppo_dispatcher.py)
  Step 4: Final results aggregation    (modules/results_aggregator.py)

Usage:
  python main.py              # full run
  python main.py --fast       # reduced PPO timesteps for quick test
  python main.py --step 1     # run only step 1
"""

import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

BASE          = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(BASE, "data", "raw")
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
RESULTS_DIR   = os.path.join(BASE, "results")
PLOTS_DIR     = os.path.join(BASE, "plots")

os.makedirs(RAW_DIR,       exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR,     exist_ok=True)
os.chdir(BASE)


def step0_prepare_data():
    print("\n" + "="*55)
    print("  STEP 0: Preparing real Meituan data")
    print("="*55)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prepare_meituan_data",
        os.path.join(BASE, "data", "prepare_meituan_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def step1_hmm():
    print("\n" + "="*55)
    print("  STEP 1: HMM Regime Detector (Person 1)")
    print("="*55)
    sys.path.insert(0, os.path.join(BASE, "modules"))
    import hmm_regime_detector as m1
    m1.run(obs_path   =os.path.join(PROCESSED_DIR, "observations.csv"),
           results_dir=RESULTS_DIR,
           plots_dir  =PLOTS_DIR)


def step2_csp():
    print("\n" + "="*55)
    print("  STEP 2: CSP + A* Assignment Engine (Person 2)")
    print("="*55)
    import csp_assignment as m2
    m2.run(orders_path  =os.path.join(PROCESSED_DIR, "orders.csv"),
           couriers_path=os.path.join(PROCESSED_DIR, "couriers.csv"),
           beliefs_path =os.path.join(RESULTS_DIR, "hmm_beliefs.csv"),
           results_dir  =RESULTS_DIR,
           plots_dir    =PLOTS_DIR)


def step3_ppo(fast=False):
    print("\n" + "="*55)
    print("  STEP 3: PPO Regime-Conditioned Dispatcher (Person 3)")
    print("="*55)
    import ppo_dispatcher as m3
    timesteps = 15_000 if fast else 40_000
    m3.run(orders_path    =os.path.join(PROCESSED_DIR, "orders.csv"),
           couriers_path  =os.path.join(PROCESSED_DIR, "couriers.csv"),
           beliefs_path   =os.path.join(RESULTS_DIR, "hmm_beliefs.csv"),
           results_dir    =RESULTS_DIR,
           plots_dir      =PLOTS_DIR,
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
║  Dataset: Real Meituan INFORMS TSL (district 3)      ║
╚══════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Reduce PPO timesteps for quick test run")
    parser.add_argument("--step", type=int, default=0,
                        help="Run only a specific step (0-4). 0=all")
    args = parser.parse_args()

    print_banner()
    t_start = time.time()

    if args.step == 0 or args.step == 0:
        step0_prepare_data()
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