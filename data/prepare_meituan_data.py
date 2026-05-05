"""
prepare_meituan_data.py
=======================
Converts raw Meituan dataset files into the exact format
expected by the project modules (HMM, CSP, PPO).

Input files (put these in your data/ folder):
  all_waybill_info_meituan_0322.csv    <- main orders file (unzipped)
  courier_wave_info_meituan.csv        <- courier waves file

Output files (saved to data/ folder):
  orders.csv        -> used by PPO dispatcher (your module)
  couriers.csv      -> used by PPO dispatcher (your module)
  observations.csv  -> used by HMM regime detector (Person 1)
  regime_truth.csv  -> used for HMM evaluation

Usage:
  python prepare_meituan_data.py
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy
import os

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_WAYBILLS = "data/all_waybill_info_meituan_0322.csv"
RAW_WAVES    = "data/courier_wave_info_meituan.csv"

OUT_ORDERS   = "data/orders.csv"
OUT_COURIERS = "data/couriers.csv"
OUT_OBS      = "data/observations.csv"
OUT_TRUTH    = "data/regime_truth.csv"

# Map real dates to day numbers (0-7)
DAY_MAP = {
    20221017: 0, 20221018: 1, 20221019: 2, 20221020: 3,
    20221021: 4, 20221022: 5, 20221023: 6, 20221024: 7
}

TZ_OFFSET_S = 8 * 3600  # UTC -> Beijing local time
COURIER_CAP = 3          # max orders per courier (from paper)
N_ZONES     = 10         # number of zones to divide city into
SIM_MINUTES = 60 * 24   # minutes in a day


# ── Helper: label regime from minute of day ────────────────────────────────────

def label_regime(minute):
    """
    Rule-based regime labelling based on temporal patterns
    described in the Meituan paper (Figure 5):
      Lunch Rush:  10:45 - 13:00
      Dinner Peak: 17:00 - 19:00
      Afternoon:   13:00 - 17:00
      Quiet:       everything else
    """
    hour = minute / 60
    if 10.75 <= hour < 13.0:
        return 1  # Lunch Rush
    elif 17.0 <= hour < 19.0:
        return 3  # Dinner Peak
    elif 13.0 <= hour < 17.0:
        return 2  # Afternoon
    else:
        return 0  # Quiet


# ── Helper: convert Unix timestamp to local minute of day ─────────────────────

def to_local_minute(unix_series):
    """Convert Unix timestamp (seconds) to Beijing local minute of day."""
    local_s = unix_series + TZ_OFFSET_S
    return ((local_s // 60) % SIM_MINUTES).astype(int)


# ── Helper: haversine distance ─────────────────────────────────────────────────

def haversine_km(lat1, lng1, lat2, lng2):
    """Vectorised haversine distance in kilometres."""
    R = 6371.0
    lat1 = np.radians(lat1)
    lng1 = np.radians(lng1)
    lat2 = np.radians(lat2)
    lng2 = np.radians(lng2)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ── Helper: assign zone from coordinates ──────────────────────────────────────

def assign_zone(lat, lng, n_zones=N_ZONES):
    """
    Divide the city into a grid and assign each location to a zone (0 to N_ZONES-1).
    Uses quantile-based binning so zones have roughly equal numbers of points.
    """
    n_bins = int(np.sqrt(n_zones))
    lat_zone = pd.qcut(lat, q=n_bins, labels=False, duplicates='drop')
    lng_zone = pd.qcut(lng, q=n_bins, labels=False, duplicates='drop')
    zone = (lat_zone * n_bins + lng_zone) % n_zones
    return zone.fillna(0).astype(int)


# ── Step 1: Build orders.csv ──────────────────────────────────────────────────

def build_orders():
    print("=" * 55)
    print("STEP 1: Building orders.csv")
    print("=" * 55)

    print(f"  Loading {RAW_WAYBILLS} ...")
    df = pd.read_csv(RAW_WAYBILLS)
    print(f"  Raw waybills loaded: {len(df):,} rows")

    # Drop unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Map date string to day number
    df['day'] = df['dt'].astype(int).map(DAY_MAP)

    # Filter: on-demand only, accepted by courier, valid day
    sub = df[
        (df['day'].notna()) &
        (df['da_id'] == 3) &
        (df['is_prebook'] == 0) &
        (df['is_courier_grabbed'] == 1)
    ].copy()

    print(f"  After filtering (on-demand, accepted): {len(sub):,} rows")

    # Convert coordinates from shifted integers to degrees
    for col in ['sender_lat', 'sender_lng', 'recipient_lat', 'recipient_lng']:
        sub[col] = sub[col] / 1e6

    # Convert timestamps to Beijing local minute of day
    sub['minute'] = to_local_minute(sub['platform_order_time'])
    sub['tw_end'] = to_local_minute(sub['estimate_arrived_time'])

    # Fix midnight wrap: if deadline is before arrival minute, add 1440
    wrap_mask = sub['tw_end'] < sub['minute']
    sub.loc[wrap_mask, 'tw_end'] = sub.loc[wrap_mask, 'tw_end'] + SIM_MINUTES
    print(f"  Midnight-wrap corrections: {wrap_mask.sum()}")

    # Calculate delivery distance (sender=merchant, recipient=customer)
    sub['distance_km'] = haversine_km(
        sub['sender_lat'], sub['sender_lng'],
        sub['recipient_lat'], sub['recipient_lng']
    ).round(3)

    # Assign zones based on customer (recipient) location
    sub['zone'] = assign_zone(sub['recipient_lat'], sub['recipient_lng'])

    # Add regime label based on time of day
    sub['regime'] = sub['minute'].apply(label_regime)

    # Time window starts when order arrives
    sub['tw_start'] = sub['minute']

    # Build clean output with required columns only
    out = sub[[
        'order_id', 'day', 'minute', 'distance_km',
        'zone', 'tw_start', 'tw_end', 'regime',
        'sender_lat', 'sender_lng', 'recipient_lat', 'recipient_lng'
    ]].copy()

    out['day'] = out['day'].astype(int)
    out = out.sort_values(['day', 'minute']).reset_index(drop=True)
    out.to_csv(OUT_ORDERS, index=False)

    print(f"\n  Orders summary:")
    print(f"    Total orders:    {len(out):,}")
    print(f"    Days covered:    {sorted(out['day'].unique().tolist())}")
    print(f"    Avg distance:    {out['distance_km'].mean():.2f} km")
    print(f"    Avg time window: {(out['tw_end'] - out['minute']).mean():.1f} min")
    print(f"    Regime distribution:")
    regime_names = {0: 'Quiet', 1: 'Lunch Rush', 2: 'Afternoon', 3: 'Dinner Peak'}
    for rid, name in regime_names.items():
        count = (out['regime'] == rid).sum()
        print(f"      {name}: {count:,}")
    print(f"\n  Saved: {OUT_ORDERS}")

    return out, sub


# ── Step 2: Build couriers.csv ────────────────────────────────────────────────

def build_couriers(orders_sub):
    print("\n" + "=" * 55)
    print("STEP 2: Building couriers.csv")
    print("=" * 55)

    # NOTE: wave_start_time is unreliable per supplementary doc section 2.2.
    # We use grab_time from waybills to get correct shift start per courier.

    # Earliest grab_time per (courier, day) = shift start
    grab_times = orders_sub[orders_sub['grab_time'] > 0].copy()
    grab_times['day'] = grab_times['day'].astype(int)
    grab_times['grab_minute'] = to_local_minute(grab_times['grab_time'])

    first_grab = (
        grab_times
        .groupby(['day', 'courier_id'])['grab_minute']
        .min()
        .reset_index()
        .rename(columns={'grab_minute': 'start_min'})
    )

    # Latest arrive_time per (courier, day) = shift end
    arrive_times = orders_sub[orders_sub['arrive_time'] > 0].copy()
    arrive_times['day'] = arrive_times['day'].astype(int)
    arrive_times['arrive_minute'] = to_local_minute(arrive_times['arrive_time'])

    last_arrive = (
        arrive_times
        .groupby(['day', 'courier_id'])['arrive_minute']
        .max()
        .reset_index()
        .rename(columns={'arrive_minute': 'end_min'})
    )

    # Merge to get full shift window per courier per day
    shifts = first_grab.merge(last_arrive, on=['day', 'courier_id'], how='inner')

    # Fix midnight wrap
    wrap = shifts['end_min'] < shifts['start_min']
    shifts.loc[wrap, 'end_min'] = SIM_MINUTES - 1
    print(f"  Midnight-wrap corrections in shifts: {wrap.sum()}")

    # Get initial courier position from their first grab location
    init_pos = orders_sub[
        (orders_sub['grab_lat'] > 0) &
        (orders_sub['grab_lng'] > 0)
    ].copy()
    init_pos['day'] = init_pos['day'].astype(int)
    init_pos['init_lat'] = init_pos['grab_lat'] / 1e6
    init_pos['init_lng'] = init_pos['grab_lng'] / 1e6

    first_pos = (
        init_pos
        .sort_values(['day', 'courier_id', 'grab_time'])
        .groupby(['day', 'courier_id'])
        .first()
        .reset_index()
        [['day', 'courier_id', 'init_lat', 'init_lng']]
    )

    # Merge position into shifts
    couriers = shifts.merge(first_pos, on=['day', 'courier_id'], how='left')

    # Assign home_zone from initial courier position
    valid = couriers[couriers['init_lat'].notna()].copy()
    couriers.loc[valid.index, 'home_zone'] = assign_zone(
        valid['init_lat'],
        valid['init_lng']
    )
    couriers['home_zone'] = couriers['home_zone'].fillna(0).astype(int)
    couriers['capacity'] = COURIER_CAP

    # Final output columns
    out = couriers[[
        'courier_id', 'day', 'start_min', 'end_min', 'home_zone',
        'init_lat', 'init_lng', 'capacity'
    ]].copy()

    out = out.sort_values(['day', 'courier_id']).reset_index(drop=True)
    out.to_csv(OUT_COURIERS, index=False)

    print(f"\n  Couriers summary:")
    print(f"    Total (courier, day) records: {len(out):,}")
    print(f"    Unique couriers: {out['courier_id'].nunique():,}")
    print(f"    Couriers per day:")
    print(out.groupby('day').size().to_string())
    avg_shift = (out['end_min'] - out['start_min']).mean()
    print(f"    Avg shift duration: {avg_shift:.0f} min ({avg_shift/60:.1f} hours)")
    print(f"\n  Saved: {OUT_COURIERS}")

    return out


# ── Step 3: Build observations.csv and regime_truth.csv ──────────────────────

def build_observations(orders_df):
    print("\n" + "=" * 55)
    print("STEP 3: Building observations.csv and regime_truth.csv")
    print("=" * 55)

    print("  Aggregating per minute (fast mode)...")

    # Count orders and average distance per day per minute
    grouped  = orders_df.groupby(['day', 'minute'])
    counts   = grouped.size().rename('orders_per_min')
    avg_dist = grouped['distance_km'].mean().rename('avg_distance')

    obs_df = pd.concat([counts, avg_dist], axis=1).reset_index()

    # Fill in missing minutes with zeros
    days = sorted(orders_df['day'].unique())
    full_index = pd.MultiIndex.from_product(
        [days, range(SIM_MINUTES)], names=['day', 'minute']
    )
    obs_df = (obs_df.set_index(['day', 'minute'])
                    .reindex(full_index, fill_value=0)
                    .reset_index())

    obs_df['avg_distance'] = obs_df['avg_distance'].fillna(0.0).round(3)
    obs_df['zone_entropy'] = 0.0
    obs_df['regime']       = obs_df['minute'].apply(label_regime)

    obs_df.to_csv(OUT_OBS, index=False)
    print(f"\n  observations.csv: {len(obs_df):,} rows")
    print(f"  Saved: {OUT_OBS}")

    truth_df = obs_df[['day', 'minute', 'regime']].rename(
        columns={'regime': 'true_regime'}
    )
    truth_df.to_csv(OUT_TRUTH, index=False)
    print(f"  Saved: {OUT_TRUTH}")

    print(f"\n  Regime distribution across all minutes:")
    regime_names = {0: 'Quiet', 1: 'Lunch Rush', 2: 'Afternoon', 3: 'Dinner Peak'}
    for rid, name in regime_names.items():
        count = (obs_df['regime'] == rid).sum()
        pct   = count / len(obs_df) * 100
        print(f"    {name}: {count:,} minutes ({pct:.1f}%)")

    return obs_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("""
╔══════════════════════════════════════════════════════╗
║  MEITUAN DATA PREPARATION                            ║
║  Converts raw dataset to project format              ║
╚══════════════════════════════════════════════════════╝
    """)

    os.makedirs("data", exist_ok=True)

    # Check input files exist before starting
    missing = []
    for f in [RAW_WAYBILLS]:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        print("ERROR: The following files were not found:")
        for f in missing:
            print(f"  {f}")
        print("\nMake sure your raw data files are in the data/ folder.")
        return

    # Run all 3 steps
    orders_clean, orders_raw = build_orders()
    couriers = build_couriers(orders_raw)
    obs = build_observations(orders_clean)

    print("\n" + "=" * 55)
    print("  DATA PREPARATION COMPLETE")
    print("=" * 55)
    print(f"  orders.csv       -> {len(orders_clean):,} orders")
    print(f"  couriers.csv     -> {len(couriers):,} courier-day records")
    print(f"  observations.csv -> {len(obs):,} minute-level records")
    print(f"  regime_truth.csv -> {len(obs):,} minute-level records")
    print("\n  You can now run the full pipeline:")
    print("  python main.py")


if __name__ == "__main__":
    main()