# Regime-Aware Courier Dispatching
### HMM + Constrained A* + Regime-Conditioned PPO

A food delivery dispatching system that infers hidden demand regimes (Quiet, Lunch Rush, Afternoon, Dinner Peak) from the live order stream using a Hidden Markov Model, and feeds that uncertainty signal directly into a PPO dispatcher and a CSP + A* assignment engine.

**Novel contribution:** Every existing dispatcher either ignores demand modes or uses raw time-of-day as a number. This system infers the hidden demand regime online via a Bayesian forward filter and feeds the full posterior distribution, not just a label, into the RL dispatcher's state space.

---

## Results

### CSP + A* Assignment

| Method | On-Time Rate | With 8-min buffer | Expired Orders |
|---|---|---|---|
| Greedy baseline | 78.4% | 97.2% | 617 |
| A* Fixed weights | 85.5% | 98.1% | 610 |
| A* + Regime weights | **85.3%** | **98.1%** | **568** |

**Per-regime on-time rate:**

| Regime | Greedy | A* Fixed | A* Regime | Winner |
|---|---|---|---|---|
| Quiet | 95.6% | 98.8% | **99.0%** | Regime ✅ |
| Lunch Rush | 62.5% | 70.4% | 69.5% | Fixed ✅ |
| Afternoon | 85.3% | 95.1% | 94.9% | Fixed ✅ |
| Dinner Peak | 68.0% | 75.8% | **76.3%** | Regime ✅ |

### PPO Dispatcher

| Method | On-Time Rate | Orders on time |
|---|---|---|
| Greedy baseline | 50.0% | 3/6 |
| PPO No Regime | 89.2% | 66/74 |
| PPO + Regime Belief | **89.7%** | 26/29 |

### Key findings
- HMM regime detection accuracy: **47%** on real Meituan data (75% confidence when correct)
- A* beats greedy by **7%** overall
- Regime-aware A* expires **49 fewer orders** than greedy
- Regime-aware A* wins on **3 out of 4 demand regimes**
- PPO + Regime Belief beats PPO without regime by **0.5%**

---

## Project Structure

```
regime-aware-courier-dispatching/
│
├── data/
│   ├── raw/                          ← Raw Meituan dataset files (not in git)
│   │   ├── all_waybill_info_meituan_0322.csv
│   │   └── courier_wave_info_meituan.csv
│   ├── processed/                    ← Output of prepare_meituan_data.py (not in git)
│   │   ├── orders.csv
│   │   ├── couriers.csv
│   │   ├── observations.csv
│   │   └── regime_truth.csv
│   └── prepare_meituan_data.py       ← Step 0: data preparation script
│
├── modules/
│   ├── hmm_regime_detector.py        ← Module 1: Bayesian forward filter
│   ├── csp_assignment.py             ← Module 2: CSP + A* dispatcher
│   ├── ppo_dispatcher.py             ← Module 3: PPO regime-conditioned agent
│   └── results_aggregator.py         ← Module 4: final plots and summary
│
├── results/                          ← Output CSVs from pipeline run
├── plots/                            ← Output plots from pipeline run
├── main.py                           ← Run the full pipeline
└── requirements.txt
```

---

## Dataset

This project uses the **Meituan INFORMS TSL 2024 Research Challenge** dataset — 654,343 real food delivery orders from a Chinese city between October 17–24, 2022.

- Download: [Meituan-INFORMS-TSL-Research-Challenge](https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge)
- Place the raw files in `data/raw/`
- The pipeline filters to **district 3, days 6–7** (Oct 23–24) for evaluation — 5,779 orders, 95 couriers
- Days 0–5 are used to train the HMM regime detector

---

## AI Topics Covered

| Module | AI Topics |
|---|---|
| HMM Regime Detector | Hidden Markov Model, Bayesian Inference, Probabilistic Reasoning over Time |
| CSP + A* Assignment | Constraint Satisfaction, A* Search, Adversarial Search |
| PPO Dispatcher | MDP, Reinforcement Learning, Sequential Decision Making |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/AmmarAlmzayek/regime-aware-courier-dispatching.git
cd regime-aware-courier-dispatching
```

**2. Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add raw data files**

Download the Meituan dataset and place these two files in `data/raw/`:
```
data/raw/all_waybill_info_meituan_0322.csv
data/raw/courier_wave_info_meituan.csv
```

---

## Running the Pipeline

**Full pipeline:**
```bash
python main.py
```

**Quick test (fewer PPO training steps):**
```bash
python main.py --fast
```

**Run a specific step only:**
```bash
python main.py --step 0   # data preparation
python main.py --step 1   # HMM regime detector
python main.py --step 2   # CSP + A* assignment
python main.py --step 3   # PPO dispatcher
python main.py --step 4   # results aggregation
```

---

## Pipeline Flow

```
data/raw/  →  prepare_meituan_data.py
                      ↓
              orders.csv, couriers.csv, observations.csv
                      ↓
              hmm_regime_detector.py  →  hmm_beliefs.csv
                      ↓
              csp_assignment.py       →  csp_metrics.csv
                      ↓
              ppo_dispatcher.py       →  ppo_evaluation.csv
                      ↓
              results_aggregator.py   →  final_summary.png
```

---

## Team

Built as a course project for Advanced AI — American University of Sharjah.

- Ammar Almzayek
- Ibrahim Gouda Mohamed
- Selam Tekleab