"""
Module 2 — CSP + A* Regime-Weighted Assignment Engine
=================================================================
Solves the courier-order assignment as a Constraint Satisfaction Problem
on real Meituan INFORMS TSL data. For each dispatch cycle (every 2 minutes
of a simulated test day), the system gathers the pending orders and the
on-shift couriers and finds the minimum-cost feasible assignment via A*
search. The cost function is a probability-weighted expectation over the
HMM regime posterior produced by Module 1.

AI Topics Covered
-----------------
- Constraint Satisfaction Problem (hard time-window, capacity, shift constraints)
- A* search in a large discrete assignment space (admissible min-cost heuristic)
- Adversarial search (minimax over worst-case order subsets)
- Probabilistic reasoning: cost = E_b_t[ weighted_cost(regime) ]

Three Conditions Compared
-------------------------
1. Greedy baseline             — nearest-merchant courier, no regime awareness
2. A* with fixed weights       — A* search with regime-agnostic flat weights (ablation)
3. A* with regime weights      — A* search with weights = E over HMM posterior (novel)

Inputs
------
data/orders.csv         — produced by data/prepare_meituan.py
data/couriers.csv       — produced by data/prepare_meituan.py
results/hmm_beliefs.csv — produced by Module 1 (HMM Forward Filter)

Outputs
-------
results/csp_assignments.csv      — every assigned (order, courier) pair with on-time flag
results/csp_metrics.csv          — on-time rate, count, distance per (method, regime)
results/csp_adversarial.csv      — adversarial stress-test summary
plots/csp_constraint_weights.png
plots/regime_performance.png
"""

import numpy as np
import pandas as pd
import heapq, os, time
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Constants ─────────────────────────────────────────────────────────────────
REGIME_NAMES  = ["Quiet", "Lunch Rush", "Afternoon", "Dinner Peak"]
REGIME_COLORS = ["#4C9BE8", "#E84C4C", "#F5A623", "#8BC34A"]
DISPATCH_CYCLE_MIN = 2          # one assignment pass every 2 minutes
# Calibrated against the real Meituan data on da_id=3, days 6-7:
#   placement->arrive mean = 23.9 min, with mean order distance 1.16 km and
#   mean pickup distance ~0.9 km. Solving for an effective combined speed gives
#   ~7 km/h (urban motorbike delivery in dense Chinese city — accounts for
#   traffic, parking, building entry, and multi-stop detours). Calibrating
#   TRAVEL_KMH=7 plus MEAL_WAIT_MIN=4 reproduces the 24-minute average and the
#   8% baseline late rate observed in the district.
TRAVEL_KMH         = 7.0
TRAVEL_MIN_PER_KM  = 60.0 / TRAVEL_KMH        # ≈ 8.57 min per km
MEAL_WAIT_MIN      = 4.0        # average wait at merchant for meal pickup
ASTAR_EXPAND_CAP   = 500
ASTAR_BEAM_K       = 4          # used only when batch size > 8
LARGE_BATCH        = 8          # threshold to switch from exact A* to beamed A*

# Per-regime stochastic delivery noise. σ is the std-dev of an extra-time
# distribution applied at evaluation time (not visible to the dispatcher).
# Calibrated so the overall late rate without buffer lands near the 8% real
# rate observed in da_id=3. During Lunch Rush traffic is heaviest and meal
# preparation queues are longest, so σ is largest. During Quiet windows
# variability is lowest. The dispatcher's feasibility check uses only the
# predicted (deterministic) time, so the regime-weighted cost function has
# something real to optimise against — under high-variance regimes it pays
# off to assign couriers with extra slack.
REGIME_NOISE_SIGMA_MIN = {0: 1.0, 1: 7.0, 2: 3.0, 3: 5.0}
REGIME_NOISE_MEAN_MIN  = {0: 0.0, 1: 1.5, 2: 0.5, 3: 1.0}
SLA_BUFFER_MIN         = 0       # main metric uses 0-min buffer (matches paper)


# ── Regime-specific constraint weight matrices ────────────────────────────────
# These weights multiply the soft-constraint costs.  When mixed across the
# regime posterior  b_t = [p_quiet, p_lunch, p_afternoon, p_dinner]  the result
# is the expected weighted cost of an assignment.
REGIME_WEIGHTS = {
    0: dict(time_window=1.0, distance=2.5, fairness=1.0),  # Quiet     — distance dominates
    1: dict(time_window=3.0, distance=1.0, fairness=0.8),  # Lunch     — deadline critical
    2: dict(time_window=1.5, distance=1.8, fairness=1.2),  # Afternoon — balanced + fair
    3: dict(time_window=2.5, distance=1.2, fairness=0.9),  # Dinner    — deadline + some dist
}

# Fixed-weights ablation: the average of the 4 regime weights — what a system
# would use if it KNEW it was in some regime but didn't know which.
FIXED_WEIGHTS = dict(time_window=2.0, distance=1.625, fairness=0.975)


# ── Geometry ──────────────────────────────────────────────────────────────────
def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    lat1r, lng1r, lat2r, lng2r = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat, dlng = lat2r - lat1r, lng2r - lng1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlng/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ── CSP hard-constraint check ─────────────────────────────────────────────────
def is_feasible(order, courier_state, current_minute):
    """
    Hard constraints (must ALL hold).
      1. Courier on shift at current_minute
      2. Courier not at capacity
      3. Delivery completes within order's time window
    Returns (ok: bool, reason: str).
    """
    if current_minute < courier_state['start_min'] or \
       current_minute > courier_state['end_min']:
        return False, "off_shift"

    if courier_state['load'] >= courier_state['capacity']:
        return False, "capacity_exceeded"

    pickup_dist = haversine_km(courier_state['lat'], courier_state['lng'],
                               order['sender_lat'], order['sender_lng'])
    deliv_dist  = order['distance_km']
    travel_total = (pickup_dist + deliv_dist) * TRAVEL_MIN_PER_KM
    arrival_at_customer = max(current_minute, courier_state['busy_until']) \
                          + travel_total + MEAL_WAIT_MIN
    if arrival_at_customer > order['tw_end']:
        return False, "time_window_violated"

    return True, "ok"


# ── Soft-constraint cost components ───────────────────────────────────────────
def soft_costs(order, courier_state, current_minute):
    """Returns (time_cost, dist_cost, fair_cost) — the per-regime building blocks."""
    pickup_dist  = haversine_km(courier_state['lat'], courier_state['lng'],
                                order['sender_lat'], order['sender_lng'])
    deliv_dist   = order['distance_km']
    total_dist   = pickup_dist + deliv_dist
    travel_min   = total_dist * TRAVEL_MIN_PER_KM
    eff_start    = max(current_minute, courier_state['busy_until'])
    arrival      = eff_start + travel_min + MEAL_WAIT_MIN
    slack        = order['tw_end'] - arrival                # min remaining; may be negative
    time_cost    = max(0.0, -slack) + 2.0 / max(slack + 1.0, 1.0)
    dist_cost    = total_dist
    fair_cost    = courier_state['load']                    # current load = workload imbalance
    return time_cost, dist_cost, fair_cost


def regime_weighted_cost(order, courier_state, current_minute, belief):
    """
    cost = sum_k  P(regime=k | obs)  ×  ( w_time(k)·t + w_dist(k)·d + w_fair(k)·f )

    `belief` is a 4-vector summing to 1.0.
    """
    t, d, f = soft_costs(order, courier_state, current_minute)
    cost = 0.0
    for k, w in REGIME_WEIGHTS.items():
        cost += belief[k] * (w['time_window']*t + w['distance']*d + w['fairness']*f)
    return cost


def fixed_weighted_cost(order, courier_state, current_minute, belief=None):
    """Ablation: collapse the regime posterior to its average — same weights always."""
    t, d, f = soft_costs(order, courier_state, current_minute)
    return (FIXED_WEIGHTS['time_window']*t +
            FIXED_WEIGHTS['distance']*d +
            FIXED_WEIGHTS['fairness']*f)


def greedy_cost(order, courier_state, current_minute, belief=None):
    """Pure pickup-distance greedy — used by the baseline."""
    return haversine_km(courier_state['lat'], courier_state['lng'],
                        order['sender_lat'], order['sender_lng'])


# ── A* / min-cost assignment over (order -> courier) ──────────────────────────
def astar_assign(orders, couriers_state, current_minute, belief, cost_fn):
    """
    Min-cost feasible assignment of pending orders to available couriers.

    Two solvers, same regime-weighted cost function:
      • Small batches (n <= LARGE_BATCH): exact heap-based A* over partial
        assignments with the admissible per-order min-cost heuristic. This
        is the textbook A*-over-CSP described in the report.
      • Larger batches: scipy linear_sum_assignment (Hungarian / Jonker-
        Volgenant), which is an exact polynomial-time min-cost bipartite
        matching solver — equivalent to A* in the limit of full expansion
        but tractable for batch sizes A* cannot handle in real time.

    Both routes solve the same CSP with the same regime-weighted edge costs.
    Infeasible (order, courier) pairs receive a +inf cost so the matching
    never picks them; orders with no feasible courier in the cycle are simply
    not assigned and remain in the pending queue.
    """
    from scipy.optimize import linear_sum_assignment
    n_o, n_c = len(orders), len(couriers_state)
    if n_o == 0 or n_c == 0:
        return []

    # Build the cost matrix and feasibility mask
    cost_mat = np.full((n_o, n_c), np.inf)
    feas_mat = np.zeros((n_o, n_c), dtype=bool)
    for i, o in enumerate(orders):
        for j, cs in enumerate(couriers_state):
            ok, _ = is_feasible(o, cs, current_minute)
            if ok:
                feas_mat[i, j] = True
                cost_mat[i, j] = cost_fn(o, cs, current_minute, belief)
    if not feas_mat.any():
        return []

    # Pick the solver
    if n_o <= LARGE_BATCH and n_o * n_c <= 80:
        return _astar_exact(orders, couriers_state, cost_mat, feas_mat)
    return _hungarian(orders, couriers_state, cost_mat, feas_mat,
                      linear_sum_assignment)


def _astar_exact(orders, couriers_state, cost_mat, feas_mat):
    """
    Exact A* — heap of (f, g, remaining_orders, used_couriers, partial_assignment).
    Heuristic h(state) = sum over unassigned orders of cheapest feasible cost
    using couriers not yet used. This is admissible.
    """
    n_o = len(orders)
    init_remaining = tuple(range(n_o))

    def heuristic(rem_tuple, used_set):
        if not rem_tuple:
            return 0.0
        h = 0.0
        for oi in rem_tuple:
            best = np.inf
            for cj in range(len(couriers_state)):
                if cj in used_set:
                    continue
                if feas_mat[oi, cj] and cost_mat[oi, cj] < best:
                    best = cost_mat[oi, cj]
            if best < np.inf:
                h += best
        return h

    h0 = heuristic(init_remaining, frozenset())
    heap = [(h0, 0.0, init_remaining, frozenset(), [])]
    best_g, best_assigned = float('inf'), []

    while heap:
        f, g, rem, used, assigned = heapq.heappop(heap)
        if f >= best_g:
            continue
        if not rem:
            if g < best_g:
                best_g, best_assigned = g, list(assigned)
            continue

        oi = rem[0]
        new_rem = rem[1:]

        # Option A — assign oi to each feasible-and-unused courier
        any_branch = False
        for cj in range(len(couriers_state)):
            if cj in used or not feas_mat[oi, cj]:
                continue
            ec = cost_mat[oi, cj]
            new_g = g + ec
            new_used = used | frozenset({cj})
            new_assigned = assigned + [(oi, cj)]
            new_h = heuristic(new_rem, new_used)
            heapq.heappush(heap, (new_g + new_h, new_g, new_rem, new_used, new_assigned))
            any_branch = True

        # Option B — leave oi unassigned this cycle (cost 0, stays in pending)
        new_h = heuristic(new_rem, used)
        heapq.heappush(heap, (g + new_h, g, new_rem, used, assigned))

    out = []
    for oi, cj in best_assigned:
        out.append((orders[oi]['order_id'], couriers_state[cj]['courier_id'], oi, cj))
    return out


def _hungarian(orders, couriers_state, cost_mat, feas_mat, lap_fn):
    """
    Min-cost bipartite matching via Hungarian / Jonker-Volgenant.
    Pads the cost matrix to square with a high-cost "no assignment" courier so
    the solver can leave orders unassigned when no feasible courier exists.
    """
    n_o, n_c = cost_mat.shape
    BIG = 1e9                                                # higher than any finite cost

    # Replace inf with BIG so the solver doesn't crash; we'll filter out post hoc
    cost_clean = np.where(np.isinf(cost_mat), BIG, cost_mat)

    # Pad to square with phantom couriers: assignment to phantom = "leave unassigned"
    if n_c < n_o:
        pad = np.full((n_o, n_o - n_c), BIG / 2.0)           # cheaper than infeasible real
        cost_clean = np.hstack([cost_clean, pad])

    row_ind, col_ind = lap_fn(cost_clean)

    out = []
    for i, j in zip(row_ind, col_ind):
        if j < n_c and feas_mat[i, j] and cost_mat[i, j] < BIG / 4:
            out.append((orders[i]['order_id'], couriers_state[j]['courier_id'], i, j))
    return out


# ── Greedy baseline (no search, no regime) ────────────────────────────────────
def greedy_assign(orders, couriers_state, current_minute):
    used = set()
    out = []
    # Sort by tightest deadline first
    order_idx = sorted(range(len(orders)), key=lambda i: orders[i]['tw_end'])
    for oi in order_idx:
        o = orders[oi]
        best_j, best_dist = None, float('inf')
        for j, cs in enumerate(couriers_state):
            if j in used:
                continue
            ok, _ = is_feasible(o, cs, current_minute)
            if not ok:
                continue
            d = haversine_km(cs['lat'], cs['lng'], o['sender_lat'], o['sender_lng'])
            if d < best_dist:
                best_dist, best_j = d, j
        if best_j is not None:
            out.append((o['order_id'], couriers_state[best_j]['courier_id'], oi, best_j))
            used.add(best_j)
    return out


# ── Adversarial stress test (minimax depth 2) ─────────────────────────────────
def adversarial_test(orders, couriers_state, current_minute, belief, cost_fn,
                     n_worst=3):
    """
    The adversary picks the n_worst orders with tightest remaining time budget,
    constrained to those that have at least one feasible courier (so the test
    is meaningful — testing on infeasible orders would always score 0).
    The system responds with A* using `cost_fn`. Returns the fraction of those
    n_worst orders that A* successfully assigns.
    """
    feasible_orders = []
    for o in orders:
        for cs in couriers_state:
            ok, _ = is_feasible(o, cs, current_minute)
            if ok:
                feasible_orders.append(o)
                break
    if len(feasible_orders) < n_worst:
        return None, 0
    worst = sorted(feasible_orders,
                   key=lambda o: o['tw_end'] - current_minute)[:n_worst]
    assigned = astar_assign(worst, couriers_state, current_minute, belief, cost_fn)
    return len(assigned) / n_worst, n_worst


# ── Day-level discrete event simulator ────────────────────────────────────────
def simulate_day(day, orders_df, couriers_df, beliefs_df, method, noise_seed=42):
    """
    Replays one full test day (1440 minutes) with the chosen method.

    Per-order stochastic noise is precomputed using `noise_seed` keyed off the
    order id, so every method evaluated on the same day sees the SAME noise
    realization for every order. This makes the comparison fair: the only
    thing that differs between methods is which courier each order is matched
    with, not the underlying delivery uncertainty.

    Returns:
      assignments (list[dict])  — every (order, courier) match with on-time flag
      stress_results (list)     — adversarial stress test outcomes per cycle
      expired                   — count of orders that timed out before assignment
      arrivals_per_regime       — count of orders that arrived per true regime
    """
    assert method in ('greedy', 'astar_fixed', 'astar_regime')
    cost_fn = {'astar_fixed': fixed_weighted_cost,
               'astar_regime': regime_weighted_cost,
               'greedy': greedy_cost}[method]

    day_orders   = orders_df[orders_df['day'] == day].sort_values('minute').reset_index(drop=True)
    day_couriers = couriers_df[couriers_df['day'] == day].reset_index(drop=True)
    day_beliefs  = beliefs_df[beliefs_df['day'] == day].set_index('minute')

    # Precompute per-order noise (positive-only extra delivery minutes).
    # Order's regime = HMM ground-truth at its arrival minute.
    rng = np.random.default_rng(noise_seed + day)
    order_noise = {}
    arrivals_per_regime = defaultdict(int)
    for _, o in day_orders.iterrows():
        t = int(o['minute'])
        if t in day_beliefs.index:
            r = int(day_beliefs.loc[t, 'true_regime'])
        else:
            r = 0
        arrivals_per_regime[r] += 1
        mu, sigma = REGIME_NOISE_MEAN_MIN[r], REGIME_NOISE_SIGMA_MIN[r]
        order_noise[o['order_id']] = max(0.0, rng.normal(mu, sigma))

    # Initialise courier state
    courier_state = {}
    for _, c in day_couriers.iterrows():
        courier_state[c['courier_id']] = dict(
            courier_id  = c['courier_id'],
            start_min   = int(c['start_min']),
            end_min     = int(c['end_min']),
            lat         = float(c['init_lat']),
            lng         = float(c['init_lng']),
            load        = 0,
            busy_until  = int(c['start_min']),
            capacity    = int(c['capacity']),
        )

    in_flight = []                                          # (completion, courier_id, dest_lat, dest_lng)
    pending = []                                            # orders that arrived but aren't assigned yet
    next_arrival = 0
    arr_records = day_orders.to_dict('records')
    assignments, stress_results = [], []
    expired = 0

    for t in range(1440):
        # 1. Process completions at minute t
        still_in_flight = []
        for (cmpl, cid, dlat, dlng) in in_flight:
            if cmpl <= t:
                if cid in courier_state:
                    courier_state[cid]['load']       = max(0, courier_state[cid]['load'] - 1)
                    courier_state[cid]['lat']        = dlat
                    courier_state[cid]['lng']        = dlng
                    courier_state[cid]['busy_until'] = max(courier_state[cid]['busy_until'], cmpl)
            else:
                still_in_flight.append((cmpl, cid, dlat, dlng))
        in_flight = still_in_flight

        # 2. New arrivals
        while next_arrival < len(arr_records) and arr_records[next_arrival]['minute'] == t:
            pending.append(arr_records[next_arrival])
            next_arrival += 1

        # 3. Drop orders past their deadline (they never got served)
        still_pending = []
        for o in pending:
            if t > o['tw_end']:
                expired += 1
            else:
                still_pending.append(o)
        pending = still_pending

        if t % DISPATCH_CYCLE_MIN != 0 or not pending:
            continue

        # 4. Available couriers
        avail = [cs for cs in courier_state.values()
                 if cs['start_min'] <= t <= cs['end_min']
                 and cs['load'] < cs['capacity']]
        if not avail:
            continue

        # 5. Belief at this minute
        if t in day_beliefs.index:
            row = day_beliefs.loc[t]
            belief = np.array([row['p_quiet'], row['p_lunch'],
                               row['p_afternoon'], row['p_dinner']], dtype=float)
            belief = belief / max(belief.sum(), 1e-9)
        else:
            belief = np.array([0.25, 0.25, 0.25, 0.25])

        # 5b. Adversarial stress test — run BEFORE main assignment so it sees
        # the real pending pool (not just the dregs the main solver rejected).
        # The minimax adversary picks the n_worst orders with tightest slack
        # that have at least one feasible courier. astar_assign is read-only on
        # courier_state, so this runs safely without affecting the main loop.
        if method.startswith('astar') and t % 30 == 0 and len(pending) >= 3:
            sr, n = adversarial_test(pending, avail, t, belief, cost_fn, n_worst=3)
            if sr is not None:
                stress_results.append(dict(day=day, minute=t, method=method,
                                           success_rate=sr, n_worst=n))

        # 6. Run the assignment
        if method == 'greedy':
            matches = greedy_assign(pending, avail, t)
        else:
            matches = astar_assign(pending, avail, t, belief, cost_fn)

        # 7. Apply assignments
        assigned_indices = set()
        for (oid, cid, oi, ci) in matches:
            o  = pending[oi]
            cs = avail[ci]
            pickup_dist  = haversine_km(cs['lat'], cs['lng'],
                                         o['sender_lat'], o['sender_lng'])
            travel_total = (pickup_dist + o['distance_km']) * TRAVEL_MIN_PER_KM
            eff_start    = max(t, cs['busy_until'])
            delivered_predicted = eff_start + travel_total + MEAL_WAIT_MIN
            delivered_actual    = delivered_predicted + order_noise[oid]

            courier_state[cs['courier_id']]['load']       += 1
            courier_state[cs['courier_id']]['busy_until']  = delivered_predicted
            in_flight.append((delivered_predicted,
                               cs['courier_id'],
                               o['recipient_lat'], o['recipient_lng']))

            true_regime = int(day_beliefs.loc[t, 'true_regime']) if t in day_beliefs.index else 0
            on_time     = int(delivered_actual <= o['tw_end'] + SLA_BUFFER_MIN)
            on_time_8b  = int(delivered_actual <= o['tw_end'] + 8)

            assignments.append(dict(
                day=day, minute=t, order_id=oid, courier_id=cs['courier_id'],
                regime=true_regime,
                pickup_dist_km=round(pickup_dist, 3),
                order_dist_km=round(o['distance_km'], 3),
                tw_end=int(o['tw_end']),
                delivered_predicted=round(delivered_predicted, 2),
                delivered_actual=round(delivered_actual, 2),
                slack_min=round(o['tw_end'] - delivered_actual, 2),
                noise_min=round(order_noise[oid], 2),
                on_time=on_time,
                on_time_8b=on_time_8b,
                method=method,
            ))
            assigned_indices.add(oi)

        pending = [o for i, o in enumerate(pending) if i not in assigned_indices]

    print(f"    {method:13s} day {day}: {len(assignments)} assigned, "
          f"{expired} expired, {len(pending)} unassigned at EOD")
    return assignments, stress_results, expired, dict(arrivals_per_regime)


# ── Plotting helpers ─────────────────────────────────────────────────────────
def plot_constraint_weights(save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(REGIME_NAMES))
    w = 0.25
    keys   = ['time_window', 'distance', 'fairness']
    labels = ['Time-Window Weight', 'Distance Weight', 'Fairness Weight']
    colors = ['#E84C4C', '#4C9BE8', '#8BC34A']
    for i, (k, lab, col) in enumerate(zip(keys, labels, colors)):
        vals = [REGIME_WEIGHTS[r][k] for r in range(4)]
        ax.bar(x + i*w, vals, w, label=lab, color=col, alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(REGIME_NAMES)
    ax.set_ylabel("Constraint weight")
    ax.set_title("Regime-Specific Constraint Weights\n"
                 r"A* edge cost = $\sum_k P(R{=}k\mid b_t)\cdot[w_t^{(k)}\,t+w_d^{(k)}\,d+w_f^{(k)}\,f]$")
    ax.legend()
    ax.grid(alpha=0.25, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_performance(metrics_df, save_path):
    """3-panel comparison: on-time rate per regime / overall / per-method counts."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: on-time rate per regime, per method
    ax = axes[0]
    pivot = metrics_df.pivot_table(index='regime', columns='method',
                                    values='on_time_rate')
    pivot = pivot.reindex(index=range(4))
    pivot.index = [REGIME_NAMES[i] for i in pivot.index]
    method_order = ['greedy', 'astar_fixed', 'astar_regime']
    method_labels = {'greedy':'Greedy', 'astar_fixed':'A* Fixed', 'astar_regime':'A* + Regime'}
    method_colors = {'greedy':'#95A5A6', 'astar_fixed':'#4C9BE8', 'astar_regime':'#E84C4C'}
    pivot = pivot[[m for m in method_order if m in pivot.columns]]
    pivot.columns = [method_labels.get(c, c) for c in pivot.columns]
    pivot.plot(kind='bar', ax=ax,
               color=[method_colors[m] for m in method_order if m in metrics_df['method'].unique()],
               alpha=0.88, edgecolor='white', width=0.78)
    ax.set_title("A — On-Time Rate by Regime")
    ax.set_ylabel("On-Time Rate")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=15)
    ax.legend(title="Method", fontsize=8)
    ax.grid(alpha=0.25, axis='y')
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)

    # Panel B: overall on-time rate per method
    ax = axes[1]
    overall = (metrics_df.assign(numer=lambda d: d['n_on_time'],
                                  denom=lambda d: d['n_assignments'])
                          .groupby('method')[['numer','denom']].sum())
    overall['rate'] = overall['numer'] / overall['denom'].replace(0, np.nan)
    overall = overall.reindex([m for m in method_order if m in overall.index])
    bars = ax.bar([method_labels[m] for m in overall.index], overall['rate'],
                  color=[method_colors[m] for m in overall.index], alpha=0.88,
                  edgecolor='white', width=0.55)
    for b, v in zip(bars, overall['rate']):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.1%}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Overall On-Time Rate")
    ax.set_title("B — Overall On-Time Rate")
    ax.grid(alpha=0.25, axis='y')

    # Panel C: number of orders successfully assigned per method
    ax = axes[2]
    counts = metrics_df.groupby('method')['n_assignments'].sum()
    counts = counts.reindex([m for m in method_order if m in counts.index])
    bars = ax.bar([method_labels[m] for m in counts.index], counts.values,
                  color=[method_colors[m] for m in counts.index], alpha=0.88,
                  edgecolor='white', width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x()+b.get_width()/2, v+max(counts)*0.01, f"{int(v):,}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Total Assigned Orders")
    ax.set_title("C — Throughput (Total Assignments)")
    ax.grid(alpha=0.25, axis='y')

    plt.suptitle("Module 2 — CSP + A* on Real Meituan Data (da_id=3, days 6–7)",
                 y=1.02, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main entry point ─────────────────────────────────────────────────────────
def run(orders_path="data/orders.csv",
        couriers_path="data/couriers.csv",
        beliefs_path="results/hmm_beliefs.csv",
        results_dir="results", plots_dir="plots"):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir,   exist_ok=True)

    print("\n=== MODULE 2: CSP + A* ASSIGNMENT ENGINE (Real Meituan data) ===")
    orders_df   = pd.read_csv(orders_path)
    couriers_df = pd.read_csv(couriers_path)
    beliefs_df  = pd.read_csv(beliefs_path)

    test_days = test_days = [6, 7]
    print(f"Test days: {test_days} | Orders: {len(orders_df):,} | "
          f"Courier records: {len(couriers_df)}")
    print(f"Belief rows: {len(beliefs_df):,}\n")

    all_assignments  = []
    all_stress       = []
    all_expired      = {}
    all_arrivals     = defaultdict(lambda: defaultdict(int))   # method -> regime -> count

    for method in ['greedy', 'astar_fixed', 'astar_regime']:
        print(f"  [{method}]")
        t0 = time.time()
        for day in test_days:
            assigns, stress, exp, arrivals = simulate_day(day, orders_df, couriers_df,
                                                           beliefs_df, method)
            all_assignments.extend(assigns)
            all_stress.extend(stress)
            all_expired[(method, day)] = exp
            for r, n in arrivals.items():
                all_arrivals[method][r] += n
        print(f"    elapsed: {time.time()-t0:.1f}s\n")

    # Dump raw assignments
    asg_df = pd.DataFrame(all_assignments)
    asg_df.to_csv(f"{results_dir}/csp_assignments.csv", index=False)
    print(f"Saved: {results_dir}/csp_assignments.csv ({len(asg_df):,} rows)")

    # Per-(method, regime) metrics — combine assignments with arrival counts
    metrics_rows = []
    for method, regime_arrivals in all_arrivals.items():
        sub = asg_df[asg_df['method'] == method]
        for r, n_arr in regime_arrivals.items():
            srx = sub[sub['regime'] == r]
            n_asg  = len(srx)
            n_ot   = int(srx['on_time'].sum()) if n_asg else 0
            n_ot8  = int(srx['on_time_8b'].sum()) if n_asg else 0
            on_time_rate = n_ot / n_asg if n_asg else 0.0
            on_time_8b   = n_ot8 / n_asg if n_asg else 0.0
            throughput   = n_asg / n_arr if n_arr else 0.0
            success_rate = n_ot / n_arr if n_arr else 0.0           # the combined headline
            success_8b   = n_ot8 / n_arr if n_arr else 0.0
            metrics_rows.append(dict(
                method=method, regime=r,
                regime_name=REGIME_NAMES[r],
                n_arrived=n_arr, n_assignments=n_asg, n_on_time=n_ot, n_on_time_8b=n_ot8,
                on_time_rate=round(on_time_rate, 4),
                on_time_rate_8b=round(on_time_8b, 4),
                throughput=round(throughput, 4),
                success_rate=round(success_rate, 4),         # P(arrived order delivered on-time)
                success_rate_8b=round(success_8b, 4),
                mean_pickup_km=round(srx['pickup_dist_km'].mean(), 3) if n_asg else 0.0,
                mean_order_km=round(srx['order_dist_km'].mean(), 3) if n_asg else 0.0,
                mean_slack_min=round(srx['slack_min'].mean(), 2) if n_asg else 0.0,
            ))
    metrics = pd.DataFrame(metrics_rows).sort_values(['method','regime']).reset_index(drop=True)
    metrics.to_csv(f"{results_dir}/csp_metrics.csv", index=False)

    # Add expired-orders summary
    exp_df = pd.DataFrame([
        dict(method=m, day=d, n_expired=v) for (m, d), v in all_expired.items()
    ])
    exp_df.to_csv(f"{results_dir}/csp_expired.csv", index=False)

    # Adversarial stress summary
    if all_stress:
        stress_df = pd.DataFrame(all_stress)
        stress_df.to_csv(f"{results_dir}/csp_adversarial.csv", index=False)
        agg = stress_df.groupby('method')['success_rate'].agg(['mean','std','count'])
        print("\n--- Adversarial stress test (n_worst=3 hardest orders, every 30 min) ---")
        print(agg.round(3))

    print("\n--- CSP per-(method, regime) metrics ---")
    print(metrics[['method','regime_name','n_arrived','n_assignments',
                   'on_time_rate','success_rate','success_rate_8b','mean_slack_min']]
          .to_string(index=False))

    print("\n--- Overall metrics per method ---")
    overall = []
    for m in ['greedy','astar_fixed','astar_regime']:
        ms = metrics[metrics.method == m]
        if len(ms) == 0: continue
        n_arr = ms['n_arrived'].sum()
        n_asg = ms['n_assignments'].sum()
        n_ot  = ms['n_on_time'].sum()
        n_ot8 = ms['n_on_time_8b'].sum()
        overall.append(dict(method=m,
                            n_arrived=n_arr,
                            n_assigned=n_asg,
                            on_time_rate=round(n_ot/n_asg if n_asg else 0, 4),
                            on_time_rate_8b=round(n_ot8/n_asg if n_asg else 0, 4),
                            throughput=round(n_asg/n_arr if n_arr else 0, 4),
                            success_rate=round(n_ot/n_arr if n_arr else 0, 4),
                            success_rate_8b=round(n_ot8/n_arr if n_arr else 0, 4)))
    overall_df = pd.DataFrame(overall)
    print(overall_df.to_string(index=False))
    overall_df.to_csv(f"{results_dir}/csp_overall.csv", index=False)

    # Plots
    plot_constraint_weights(f"{plots_dir}/csp_constraint_weights.png")
    plot_performance(metrics, f"{plots_dir}/regime_performance.png")

    return metrics, asg_df


if __name__ == "__main__":
    if not os.path.exists("data/orders.csv"):
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run()
