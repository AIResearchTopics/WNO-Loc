# -*- coding: utf-8 -*-
"""
Multi-Region Folder Scanner, Event Processor, and Fold Builder.

Folder structure expected:
    DATAFOLDER/
        EU/
            Berlin/
                Berlin_2022.csv          <- sensor data
                Berlin_2023.csv
                Berlin_2024.csv
                Berlin_events_2022.csv   <- event data
                Berlin_events_2023.csv
                Berlin_events_2024.csv
            Lisbon/ Munich/ Paris/ Rome/ Stockholm/
        US/
            Arizona/
                Arizona_2022.csv
                Arizona_2023.csv
                Arizona_events_2022.csv
                Arizona_events_2023.csv
            California/ Nevada/ NewMexico/ Utah/

Public API
----------
scan_and_process_folder(data_folder, window_length, ...)
    -> list of region dicts, one per (city, year)
    -> cached to .cache/ after first run

build_fold(regions, fold)
    -> (train_regions, val_regions, test_regions, labels)

Available folds
---------------
"US_TO_EUROPE"   Train: all US       Val: Paris+Munich    Test: Berlin+Rome
"EUROPE_TO_US"   Train: all EU       Val: Arizona+Nevada  Test: CA+NM+Utah
"WITHIN_EU"      Train: Munich+Rome+Lisbon+Stockholm  Val: Paris  Test: Berlin
"WITHIN_US"      Train: CA+NV+NM+UT  Val: Arizona         Test: Arizona (diff year)
"TEMPORAL"       Train: all 2022     Val: all 2023        Test: all 2024

Each region dict contains:
    U:          (n_events, N, T, F)  z-score normalized per city-year
    locations:  (n_events, 2)        event lat/lon
    coords:     (N, 2)               sensor lat/lon
    name:       str                  e.g. "Berlin"
    region:     str                  "EU" or "US"
    year:       int                  2022 / 2023 / 2024
    n_features: int                  11

@author: usman.anjum
"""

import os
import re
import hashlib
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    'PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
    'wind_speed', 'wind_direction', 'temperature',
    'pressure', 'precipitation'
]
F = len(FEATURE_COLS)

# Event types that use placeholder year-long dates
# These get peak-PM2.5 detection instead of using the CSV date
PLACEHOLDER_DATE_TYPES = {'wildfire'}

# City groupings for fold building
EU_CITIES = {
    'berlin', 'lisbon', 'munich', 'paris', 'rome', 'stockholm'
}
US_CITIES = {
    'arizona', 'california', 'nevada', 'newmexico', 'utah'
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_path(data_folder: str, window_length: int) -> str:
    h = hashlib.md5(
        f"{data_folder}|{window_length}".encode()
    ).hexdigest()[:12]
    cache_dir = os.path.join(data_folder, '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'regions_{h}.npz')


def _cache_valid(cache_file: str, data_folder: str) -> bool:
    """Cache is valid if newer than all CSVs in the data folder tree."""
    if not os.path.exists(cache_file):
        return False
    cache_mtime = os.path.getmtime(cache_file)
    for root, _, files in os.walk(data_folder):
        if '.cache' in root:
            continue
        for fname in files:
            if fname.endswith('.csv'):
                if os.path.getmtime(os.path.join(root, fname)) > cache_mtime:
                    return False
    return True


def _save_cache(cache_file: str, regions: list):
    arrays = {}
    for i, r in enumerate(regions):
        arrays[f'{i}_U']         = r['U']
        arrays[f'{i}_locations'] = r['locations']
        arrays[f'{i}_coords']    = r['coords']
        arrays[f'{i}_name']      = np.array([r['name']])
        arrays[f'{i}_region']    = np.array([r['region']])
        arrays[f'{i}_year']      = np.array([r['year']])
        arrays[f'{i}_nf']        = np.array([r['n_features']])
        if r.get('lap') is not None:
            arrays[f'{i}_lap'] = r['lap']
            arrays[f'{i}_adj'] = r['adj']
    np.savez_compressed(cache_file, **arrays)
    print(f"  Cache saved -> {cache_file}")


def _load_cache(cache_file: str) -> list:
    data    = np.load(cache_file, allow_pickle=True)
    n       = sum(1 for k in data.files if k.endswith('_name'))
    regions = []
    for i in range(n):
        regions.append({
            'U':          data[f'{i}_U'],
            'locations':  data[f'{i}_locations'],
            'coords':     data[f'{i}_coords'],
            'name':       str(data[f'{i}_name'][0]),
            'region':     str(data[f'{i}_region'][0]),
            'year':       int(data[f'{i}_year'][0]),
            'n_features': int(data[f'{i}_nf'][0]),
            'lap': data[f'{i}_lap'] if f'{i}_lap' in data.files else None,
            'adj': data[f'{i}_adj'] if f'{i}_adj' in data.files else None,
        })
    return regions


# ---------------------------------------------------------------------------
# Folder scanner
# ---------------------------------------------------------------------------
def _scan_folder(data_folder: str) -> list:
    """
    Walk EU/ and US/ subfolders.
    Returns list of registry dicts:
        {city, city_display, region, year, sensor_files, event_files}
    one entry per (city, year) that has both sensor and event files.
    """
    registry = []

    for region in ['EU', 'US']:
        region_path = os.path.join(data_folder, region)
        if not os.path.isdir(region_path):
            print(f"  [WARN] {region}/ folder not found: {region_path}")
            continue

        for city_folder in sorted(os.listdir(region_path)):
            city_path = os.path.join(region_path, city_folder)
            if not os.path.isdir(city_path):
                continue

            # Normalize city name for matching
            city_key = city_folder.lower().replace(' ', '').replace('_', '')

            sensor_by_year = {}
            events_by_year = {}

            for fname in sorted(os.listdir(city_path)):
                if not fname.endswith('.csv'):
                    continue
                fpath  = os.path.join(city_path, fname)
                flower = fname.lower()

                # Detect year from filename
                year = None
                for y in ['2022', '2023', '2024', '2025']:
                    if y in flower:
                        year = int(y)
                        break

                if 'events' in flower:
                    events_by_year.setdefault(year, []).append(fpath)
                else:
                    sensor_by_year.setdefault(year, []).append(fpath)

            # One registry entry per year with both sensor + event files
            all_years = sorted(
                set(sensor_by_year.keys()) | set(events_by_year.keys())
            )
            for yr in all_years:
                s_files = sensor_by_year.get(yr, [])
                e_files = events_by_year.get(yr, [])
                if s_files and e_files:
                    registry.append({
                        'city':         city_key,
                        'city_display': city_folder,
                        'region':       region,
                        'year':         yr,
                        'sensor_files': s_files,
                        'event_files':  e_files,
                    })
                elif s_files:
                    print(f"  [WARN] {city_folder} {yr}: "
                          f"sensor data but no events file")
                elif e_files:
                    print(f"  [WARN] {city_folder} {yr}: "
                          f"events file but no sensor data")

    return registry


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _load_sensor(sensor_files: list) -> pd.DataFrame:
    dfs = [pd.read_csv(f, low_memory=False) for f in sensor_files]
    env = pd.concat(dfs, ignore_index=True)
    env['timestamp'] = pd.to_datetime(
        env['timestamp'], utc=True
    ).dt.tz_localize(None)
    env = env.drop_duplicates(
        subset=['site_id', 'timestamp']
    ).sort_values(['site_id', 'timestamp']).reset_index(drop=True)
    return env


def _load_events(event_files: list,
                 max_tier: int = 3,
                 exclude_diffuse: bool = False,
                 max_wildfire_dist_km: float = 40.0) -> pd.DataFrame:
    """
    Load event files and apply quality filters:
        max_tier             : keep events with tier <= this (1=strong only)
        exclude_diffuse      : if True, drop diffuse=True events
        max_wildfire_dist_km : drop wildfires beyond this distance from city
    """
    dfs = [pd.read_csv(f) for f in event_files]
    ev  = pd.concat(dfs, ignore_index=True)

    ev['date'] = pd.to_datetime(
        ev['date'], utc=True, errors='coerce'
    ).dt.tz_localize(None)
    if 'end_date' in ev.columns:
        ev['end_date'] = pd.to_datetime(
            ev['end_date'], utc=True, errors='coerce'
        ).dt.tz_localize(None)

    ev = ev.dropna(subset=['lat', 'lon', 'date']).reset_index(drop=True)

    n_raw = len(ev)

    # Filter by tier
    if 'tier' in ev.columns:
        tier_num = pd.to_numeric(ev['tier'], errors='coerce').fillna(3)
        ev = ev[tier_num <= max_tier].reset_index(drop=True)

    # Filter diffuse events
    if exclude_diffuse and 'diffuse' in ev.columns:
        diffuse_mask = ev['diffuse'].astype(str).str.lower().isin(
            ['true', '1', 'yes']
        )
        ev = ev[~diffuse_mask].reset_index(drop=True)

    # Filter wildfires beyond sensor range
    if 'notes' in ev.columns:
        def _parse_dist(notes):
            if not isinstance(notes, str):
                return 0.0
            m = re.search(r'dist_km=([\d.]+)', notes)
            return float(m.group(1)) if m else 0.0

        is_wildfire = ev.get('type', pd.Series()).str.lower().isin(
            PLACEHOLDER_DATE_TYPES
        )
        dist_km = ev['notes'].apply(_parse_dist)
        far_wildfire = is_wildfire & (dist_km > max_wildfire_dist_km)
        ev = ev[~far_wildfire].reset_index(drop=True)

    n_filtered = n_raw - len(ev)
    if n_filtered > 0:
        print(f"    Filtered {n_filtered} events "
              f"(tier/diffuse/dist) -> {len(ev)} remaining")

    return ev


# ---------------------------------------------------------------------------
# Peak detection for wildfire placeholder dates
# ---------------------------------------------------------------------------
def _find_peak_onset(env_df: pd.DataFrame, year: int) -> pd.Timestamp:
    """Find peak PM2.5/PM10 timestamp for wildfire events."""
    yr_df = env_df[env_df['timestamp'].dt.year == year]
    if yr_df.empty:
        return pd.Timestamp(f'{year}-07-01')
    for col in ['PM2_5', 'PM10']:
        if col not in yr_df.columns:
            continue
        sig = pd.to_numeric(yr_df[col], errors='coerce')
        if sig.isna().all():
            continue
        tmp        = yr_df.copy()
        tmp['_s']  = sig
        ts         = tmp.groupby('timestamp')['_s'].mean()
        return ts.rolling(24, min_periods=1, center=True).mean().idxmax()
    return pd.Timestamp(f'{year}-07-01')


# ---------------------------------------------------------------------------
# Time window extraction
# ---------------------------------------------------------------------------
def _extract_window(env_df, unique_sites, site_to_idx,
                    onset, pre_hours: int, T: int) -> np.ndarray:
    N     = len(unique_sites)
    start = onset - pd.Timedelta(hours=pre_hours)
    end   = start + pd.Timedelta(hours=T - 1)

    mask = (
        (env_df['timestamp'] >= start - pd.Timedelta(hours=2)) &
        (env_df['timestamp'] <= end   + pd.Timedelta(hours=2))
    )
    wdf  = env_df[mask]
    tgt  = pd.date_range(start=start, end=end, freq='1h')
    cube = np.zeros((N, T, F), dtype=np.float32)

    for site in unique_sites:
        sdf = wdf[wdf['site_id'] == site].copy()
        if sdf.empty:
            continue
        sdf = sdf.set_index('timestamp')
        for col in FEATURE_COLS:
            if col not in sdf.columns:
                sdf[col] = np.nan
        res = sdf[FEATURE_COLS].resample('1h').mean() \
              if len(sdf) != T else sdf[FEATURE_COLS]
        res = res.reindex(tgt)
        cube[site_to_idx[site]] = np.nan_to_num(
            res.values, nan=0.0
        ).astype(np.float32)[:T]

    return cube


# ---------------------------------------------------------------------------
# Per city-year z-score normalization
# ---------------------------------------------------------------------------
def _normalize(U: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each feature across all events, sensors,
    and timesteps for this city-year.

    Replaces the old per-sensor min-max which destroyed
    cross-sensor comparability needed for P(x) neighbor contrast.
    """
    out = U.copy()
    for fi in range(U.shape[-1]):
        ch  = U[..., fi]
        std = ch.std()
        out[..., fi] = (ch - ch.mean()) / std if std > 1e-8 else 0.0
    return out


# ---------------------------------------------------------------------------
# Process one city-year entry
# ---------------------------------------------------------------------------
def _process_entry(entry: dict, T: int,
                   pre_hours: int = 48,
                   max_tier: int = 3,
                   exclude_diffuse: bool = False,
                   max_wildfire_dist_km: float = 40.0) -> dict | None:

    city   = entry['city_display']
    region = entry['region']
    year   = entry['year']

    print(f"\nProcessing Region ({region}) {city} {year}")

    env_df = _load_sensor(entry['sensor_files'])
    ev_df  = _load_events(
        entry['event_files'],
        max_tier=max_tier,
        exclude_diffuse=exclude_diffuse,
        max_wildfire_dist_km=max_wildfire_dist_km,
    )

    unique_sites = sorted(env_df['site_id'].unique())
    N            = len(unique_sites)
    site_to_idx  = {s: i for i, s in enumerate(unique_sites)}

    coords = np.array([
        [env_df[env_df['site_id'] == s]['latitude'].iloc[0],
         env_df[env_df['site_id'] == s]['longitude'].iloc[0]]
        for s in unique_sites
    ], dtype=np.float32)
    
    print(f"  Building sensor graph ({N} sensors)...", end=' ', flush=True)
    try:
        from sensor_graph import SensorGraph
        _mask = np.ones((N, F), dtype=bool)
        print(f"  Building sensor graph ({N} sensors)...", end=' ', flush=True)

    
        # Nevada has sparse feature coverage causing _bridge_components
        # to hang -- use direct adjacency without bridging for small sparse networks
        if city.lower() == 'nevada' or N <= 40:
            distance  = np.linalg.norm(
                coords[:, None, :] - coords[None, :, :], axis=2
            )
            A_feature = np.zeros((F, N, N), dtype=np.float32)
            for f in range(F):
                for i in range(N):
                    others  = np.arange(N)
                    others  = others[others != i]
                    k_eff   = min(4, len(others))
                    nearest = others[np.argsort(distance[i, others])[:k_eff]]
                    for j in nearest:
                        w = np.exp(-distance[i, j]**2 / (2 * 0.2**2))
                        A_feature[f, i, j] = w
                A_feature[f] = np.maximum(A_feature[f], A_feature[f].T)
            
            L = np.zeros_like(A_feature)
            for f in range(F):
                deg      = A_feature[f].sum(axis=1)
                inv_sqrt = np.where(deg > 0, 1.0/np.sqrt(deg.clip(min=1e-8)), 0.0)
                D_inv    = np.diag(inv_sqrt)
                L[f]     = -(D_inv @ A_feature[f] @ D_inv)
            
            lap_precomputed = L.astype(np.float32)
            adj_precomputed = A_feature.astype(np.float32)
        else:
            _sg = SensorGraph(coords, _mask, k=4, sigma=0.2)
            lap_precomputed = _sg.laplacian().astype(np.float32)
            adj_precomputed = _sg.feature_adjacency().astype(np.float32)
        
        print("done.")

    except Exception as e:
        print(f"failed ({e}) -- will build per-event.")
        lap_precomputed = None
        adj_precomputed = None
    
    data_start = env_df['timestamp'].min()
    data_end   = env_df['timestamp'].max()

    print(f"  {N} sensors | "
          f"{data_start.date()} to {data_end.date()} | "
          f"{len(ev_df)} events after filtering")

    cubes         = []
    locations     = []
    seen_wf_years = set()

    for _, ev in ev_df.iterrows():
        etype = str(ev.get('type', '')).lower()
        date  = ev['date']

        # Wildfire: deduplicate by year, use peak PM2.5 date
        if etype in PLACEHOLDER_DATE_TYPES:
            yr = date.year
            if yr in seen_wf_years:
                continue
            seen_wf_years.add(yr)
            onset = _find_peak_onset(env_df, yr)
        else:
            onset = date

        start = onset - pd.Timedelta(hours=pre_hours)
        end   = start + pd.Timedelta(hours=T - 1)

        if start < data_start or end > data_end:
            continue

        cube = _extract_window(
            env_df, unique_sites, site_to_idx, onset, pre_hours, T
        )
        if cube.sum() == 0:
            continue

        cubes.append(cube)
        locations.append([float(ev['lat']), float(ev['lon'])])

    if not cubes:
        print(f"  [SKIP] No valid events packaged for {city} {year}")
        return None

    U = _normalize(np.stack(cubes, axis=0))

    print(f"-> Successfully packaged {len(cubes)} events. "
          f"Tensor Shape: {U.shape}")

    return {
        'U':          U,
        'locations':  np.array(locations, dtype=np.float32),
        'coords':     coords,
        'lap':        lap_precomputed,   
        'adj':        adj_precomputed,   
        'name':       city,
        'region':     region,
        'year':       year,
        'n_features': F,
    }


# ---------------------------------------------------------------------------
# Public API: scan and process
# ---------------------------------------------------------------------------
def scan_and_process_folder(data_folder: str,
                             window_length: int = 100,
                             pre_event_hours: int = 48,
                             max_tier: int = 3,
                             exclude_diffuse: bool = False,
                             max_wildfire_dist_km: float = 40.0,
                             force_reprocess: bool = False) -> list:
    """
    Scan EU/ and US/ subfolders, process all city-year combinations,
    and return a list of region dicts.

    Results are cached to DATAFOLDER/.cache/ after first run and
    reloaded instantly on subsequent runs (including different seeds
    on Colab -- cache persists on Google Drive).

    Parameters
    ----------
    data_folder          : path to root data folder containing EU/ and US/
    window_length        : T -- time window length in hours (default 100)
    pre_event_hours      : hours before event onset to start window (default 48)
    max_tier             : keep events with tier <= this (default 3 = all)
    exclude_diffuse      : drop citywide diffuse events (default False)
    max_wildfire_dist_km : drop wildfires beyond this km from city (default 40)
    force_reprocess      : ignore cache and reprocess from raw CSVs

    Returns
    -------
    list of region dicts, sorted by region then city then year.
    Each dict: U, locations, coords, name, region, year, n_features
    """
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    cache_file = _cache_path(data_folder, window_length)

    if not force_reprocess and _cache_valid(cache_file, data_folder):
        print(f"\nLoading processed data from cache: {cache_file}")
        regions = _load_cache(cache_file)
        print(f"Cache hit! {len(regions)} region-year entries:")
        for r in regions:
            print(f"  [{r['region']}] {r['name']:15s} {r['year']} "
                  f"| {r['U'].shape[0]:3d} events "
                  f"| {r['coords'].shape[0]:4d} sensors "
                  f"| shape {r['U'].shape}")
        return regions

    print(f"\nScanning data folder: {data_folder}")
    registry = _scan_folder(data_folder)

    print(f"\nFound {len(registry)} city-year combinations:")
    for e in registry:
        print(f"  [{e['region']}] {e['city_display']:15s} {e['year']} "
              f"| sensors: {len(e['sensor_files'])} file(s) "
              f"| events: {len(e['event_files'])} file(s)")

    regions = []
    for entry in registry:
        result = _process_entry(
            entry,
            T             = window_length,
            pre_hours     = pre_event_hours,
            max_tier      = max_tier,
            exclude_diffuse       = exclude_diffuse,
            max_wildfire_dist_km  = max_wildfire_dist_km,
        )
        if result is not None:
            regions.append(result)

    if not regions:
        raise RuntimeError(
            "No regions processed. Check folder structure and file naming."
        )

    _save_cache(cache_file, regions)

    print(f"\n{'='*60}")
    print(f"Data pipeline complete! {len(regions)} region-year entries:")
    for r in regions:
        print(f"  [{r['region']}] {r['name']:15s} {r['year']} "
              f"| {r['U'].shape[0]:3d} events "
              f"| {r['coords'].shape[0]:4d} sensors "
              f"| shape {r['U'].shape}")
    print(f"{'='*60}")
    
    return regions


# ---------------------------------------------------------------------------
# Public API: build fold
# ---------------------------------------------------------------------------
def build_fold(regions: list, fold: str):
    
    def _select(names=None, region=None, year=None, years=None):
        out = []
        for r in regions:
            if names is not None:
                if r['name'].lower() not in [n.lower() for n in names]:
                    continue
            if region is not None:
                if r['region'] != region:
                    continue
            if year is not None:
                if r['year'] != year:
                    continue
            if years is not None:
                if r['year'] not in years:
                    continue
            out.append(r)
        return out

    fold = fold.upper()

    # ----------------------------------------------------------------
    # Fold 1: US_TO_EUROPE
    # Cross-continental transfer -- primary result
    # Train: all US cities all years (382 events)
    # Val:   Paris + Munich (52 events)
    # Test:  Berlin + Rome (46 events)
    # ----------------------------------------------------------------
    if fold == "US_TO_EUROPE":
        train = _select(region='US')
        val   = _select(names=['Paris', 'Munich'])
        test  = _select(names=['Berlin', 'Rome'])
        labels = {
            'train': 'US_All',
            'val':   'Paris_Munich_Validation',
            'test':  'Berlin_Rome_Unseen_Test',
        }

    # ----------------------------------------------------------------
    # Fold 2: WITHIN_EU
    # Within-continent spatial transfer
    # Train: Munich + Rome + Lisbon + Stockholm (93 events)
    # Val:   Paris (23 events)
    # Test:  Berlin (24 events)
    # ----------------------------------------------------------------
    elif fold == "WITHIN_EU":
        train = _select(names=['Munich', 'Rome', 'Lisbon', 'Stockholm'])
        val   = _select(names=['Paris'])
        test  = _select(names=['Berlin'])
        labels = {
            'train': 'Munich_Rome_Lisbon_Stockholm',
            'val':   'Paris_Validation',
            'test':  'Berlin_Unseen_Test',
        }

    # ----------------------------------------------------------------
    # Fold 3: WITHIN_US_TEMPORAL
    # Temporal generalization within same sensor network
    # Train: Arizona 2022 (75 events, 80 sensors)
    # Val:   Arizona 2023 (49 events, 79 sensors)
    # Test:  Arizona 2024 (54 events, 83 sensors)
    # No scale mismatch -- same city different years
    # ----------------------------------------------------------------
    elif fold == "WITHIN_US_TEMPORAL":
        train = _select(names=['Arizona'], year=2022)
        val   = _select(names=['Arizona'], year=2023)
        test  = _select(names=['Arizona'], year=2024)
        labels = {
            'train': 'Arizona_2022',
            'val':   'Arizona_2023_Validation',
            'test':  'Arizona_2024_Unseen_Test',
        }

    # ----------------------------------------------------------------
    # Fold 4: WITHIN_US_SPATIAL_S2M
    # Small → Medium scale transfer within US
    # Train: Nevada + NewMexico + Utah (69 events, 29-36 sensors)
    # Val:   Arizona 2022 (75 events, 80 sensors)
    # Test:  Arizona 2023+2024 (103 events, 79-83 sensors)
    # Scale ratio: ~2.5x (29-36 → 80 sensors)
    # ----------------------------------------------------------------
    elif fold == "WITHIN_US_SPATIAL_S2M":
        train = _select(names=['Nevada', 'NewMexico', 'Utah'])
        val   = _select(names=['Arizona'], year=2022)
        test  = _select(names=['Arizona'], years=[2023, 2024])
        labels = {
            'train': 'Nevada_NM_Utah_All',
            'val':   'Arizona_2022_Validation',
            'test':  'Arizona_2023_2024_Unseen_Test',
        }

    # ----------------------------------------------------------------
    # Fold 5: WITHIN_US_SPATIAL_M2L
    # Medium → Large scale transfer within US
    # Train: Arizona (178 events, 79-83 sensors)
    # Val:   Nevada + NewMexico + Utah (~69 events, 29-36 sensors)
    # Test:  California (135 events, 227-231 sensors)
    # Scale ratio: ~3x (80 → 227-231 sensors)
    # ----------------------------------------------------------------
    elif fold == "WITHIN_US_SPATIAL_M2L":
        train = _select(names=['Arizona'])
        val   = _select(names=['Nevada', 'NewMexico', 'Utah'])
        test  = _select(names=['California'])
        labels = {
            'train': 'Arizona_All',
            'val':   'Nevada_NM_Utah_Validation',
            'test':  'California_Unseen_Test',
        }

    elif fold == "WITHIN_US_SPATIAL_L2M":
        train = _select(names=['California'])
        val   = _select(names=['Nevada', 'NewMexico', 'Utah'])
        test  = _select(names=['Arizona'])
        labels = {
            'train': 'California_All',
            'val':   'Nevada_NM_Utah_Validation',
            'test':  'Arizona_Unseen_Test',
        }

    # ----------------------------------------------------------------
    # Fold 6: TEMPORAL_EU
    # Temporal generalization across multiple European cities
    # Train: All EU cities 2022 (~39 events)
    # Val:   All EU cities 2023 (~46 events)
    # Test:  All EU cities 2024 (~46 events)
    # Same 6 cities, different years -- tests temporal stability
    # ----------------------------------------------------------------
    elif fold == "TEMPORAL_EU":
        train = _select(region='EU', year=2022)
        val   = _select(region='EU', year=2023)
        test  = _select(region='EU', year=2024)
        labels = {
            'train': 'EU_All_2022',
            'val':   'EU_All_2023_Validation',
            'test':  'EU_All_2024_Unseen_Test',
        }

    # ----------------------------------------------------------------
    # Fold 7: EUROPE_TO_US
    # Negative result -- extreme scale mismatch
    # Train: All EU cities (140 events, 5-36 sensors, 30km coverage)
    # Val:   Arizona + Nevada (193 events, 35-83 sensors)
    # Test:  California + NewMexico + Utah (172 events, 29-231 sensors)
    # Scale ratio: ~10x+ -- expected to fail for all methods
    # ----------------------------------------------------------------
    elif fold == "EUROPE_TO_US":
        train = _select(region='EU')
        val   = _select(names=['Arizona', 'Nevada'])
        test  = _select(names=['California', 'NewMexico', 'Utah'])
        labels = {
            'train': 'EU_All',
            'val':   'Arizona_Nevada_Validation',
            'test':  'California_NM_Utah_Unseen_Test',
        }

    else:
        valid = [
            'US_TO_EUROPE', 'WITHIN_EU',
            'WITHIN_US_TEMPORAL', 'WITHIN_US_SPATIAL_S2M',
            'WITHIN_US_SPATIAL_M2L', 'WITHIN_US_SPATIAL_L2M',
            'TEMPORAL_EU', 'EUROPE_TO_US'
        ]
        raise ValueError(
            f"Unknown fold '{fold}'. Choose from: {valid}"
        )

    # Print fold summary
    print(f"\n{'='*60}")
    print(f"Fold: {fold}")

    def _summarize(label, lst):
        if not lst:
            print(f"  {label:8s}: [EMPTY]")
            return
        total  = sum(r['U'].shape[0] for r in lst)
        detail = ' | '.join(
            f"{r['name']}({r['year']}):{r['U'].shape[0]}ev"
            for r in lst
        )
        print(f"  {label:8s}: {total:4d} events -- {detail}")

    _summarize('TRAIN', train)
    _summarize('VAL',   val)
    _summarize('TEST',  test)
    print(f"{'='*60}")

    return train, val, test, labels


# ---------------------------------------------------------------------------
# Helper: convert list of region dicts to EventDataset / ConcatDataset
# This is used in main.py to avoid repeating the same loop
# ---------------------------------------------------------------------------
def regions_to_dataset(region_list: list, k_neighbors: int = 4,
                        graph_sigma: float = 0.2,
                        coverage_prob: float = 0.8,
                        mask_seed: int = 42, 
                        verbose = False):
    """
    Convert a list of region dicts into a single ConcatDataset.

    Imported and called in main.py -- keeps main.py clean.
    Returns None if region_list is empty.
    """
    from dataset import EventDataset

    if not region_list:
        return None

    datasets = []
    for data in region_list:
        ds = EventDataset(
                generator        = None,
                coords           = data['coords'],
                real_U           = data['U'],
                real_locations   = data['locations'],
                precomputed_lap  = data.get('lap'),
                precomputed_adj  = data.get('adj'),
                k_neighbors      = k_neighbors,
                graph_sigma      = graph_sigma,
                feature_mask     = None,
                coverage_prob    = coverage_prob,
                mask_seed        = mask_seed,
                verbose          = verbose,
            )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
