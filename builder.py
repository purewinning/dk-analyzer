# builder.py

import pandas as pd
import numpy as np 
from pulp import *
from typing import Dict, List, Tuple, Union, Any

# --- CONFIGURATION CONSTANTS ---
PUNT_THR = 10.0
CHALK_THR = 30.0
MEGA_CHALK_THR = 40.0

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8
DEFAULT_STD_DEV_PCT = 0.20  # 20% of projection for standard deviation in MCS


# --- HELPER FUNCTIONS FOR TIER / OWNERSHIP BUCKETS ---

def classify_ownership_bucket(ownership: float) -> str:
    """
    Classifies ownership into buckets: 'punt', 'chalk', 'mega', or 'mid'.
    """
    if ownership <= PUNT_THR:
        return "punt"
    elif ownership >= MEGA_CHALK_THR:
        return "mega"
    elif ownership >= CHALK_THR:
        return "chalk"
    else:
        return "mid"


def get_bucket_order(template_tier: str) -> List[str]:
    """
    Returns the preference order of ownership buckets based on a template tier.
    """
    tier_to_bucket_order = {
        "punt": ["punt", "mid", "chalk", "mega"],
        "mid": ["mid", "chalk", "punt", "mega"],
        "chalk": ["chalk", "mid", "mega", "punt"],
        "mega": ["mega", "chalk", "mid", "punt"],
    }
    return tier_to_bucket_order.get(template_tier, ["mid", "chalk", "punt", "mega"])


# --- TEMPLATE CLASS FOR LINEUP STRUCTURE ---

class LineupTemplate:
    """
    Represents the ownership bucket template for a lineup.
    Example: 2 mega chalk, 3 chalk, 2 mid, 1 punt, etc.
    """

    def __init__(
        self,
        num_punt: int,
        num_mid: int,
        num_chalk: int,
        num_mega: int,
        salary_cap: int = DEFAULT_SALARY_CAP,
        roster_size: int = DEFAULT_ROSTER_SIZE,
    ):
        self.num_punt = num_punt
        self.num_mid = num_mid
        self.num_chalk = num_chalk
        self.num_mega = num_mega
        self.salary_cap = salary_cap
        self.roster_size = roster_size

    def to_dict(self) -> Dict[str, int]:
        return {
            "punt": self.num_punt,
            "mid": self.num_mid,
            "chalk": self.num_chalk,
            "mega": self.num_mega,
        }

    def __repr__(self) -> str:
        return (
            f"LineupTemplate(punt={self.num_punt}, mid={self.num_mid}, "
            f"chalk={self.num_chalk}, mega={self.num_mega}, "
            f"salary_cap={self.salary_cap}, roster_size={self.roster_size})"
        )


# --- EXPOSURE / OWNERSHIP HELPERS ---

def calculate_exposure_limits(
    slate_df: pd.DataFrame,
    max_chalk_pct: float,
    max_mega_pct: float,
) -> Dict[str, float]:
    """
    Build max exposure dictionary keyed by player_id.
    """
    exposure_limits: Dict[str, float] = {}

    for _, row in slate_df.iterrows():
        pid = row["player_id"]
        own = float(row.get("own", 0.0))
        bucket = classify_ownership_bucket(own)

        if bucket == "chalk":
            exposure_limits[pid] = max_chalk_pct
        elif bucket == "mega":
            exposure_limits[pid] = max_mega_pct
        else:
            # punt / mid, no hard cap unless set elsewhere
            exposure_limits[pid] = 1.0

    return exposure_limits


def apply_manual_exposure_overrides(
    exposure_limits: Dict[str, float],
    manual_overrides: Dict[str, float],
) -> Dict[str, float]:
    """
    Apply user-specified overrides to exposure_limits.
    """
    for pid, val in manual_overrides.items():
        exposure_limits[pid] = val
    return exposure_limits


# --- LINEUP OPTIMIZER (SINGLE LINEUP VIA PULP) ---

def optimize_single_lineup(
    slate_df: pd.DataFrame, 
    template: 'LineupTemplate', 
    bucket_slack: int,
    locked_player_ids: List[str], 
    excluded_player_ids: List[str]
) -> Union[List[str], None]:
    """
    Finds the lineup that maximizes projection (or sampled projection) while 
    prioritizing template adherence. Returns a list of player_ids.
    """
    
    # Filter for playable players
    playable_df = slate_df[~slate_df['player_id'].isin(excluded_player_ids)].copy()
    
    # CRITICAL: Create mappings for safe, quick access
    proj_map = playable_df.set_index('player_id')['proj'].to_dict()
    salary_map = playable_df.set_index('player_id')['salary'].to_dict()
    own_map = playable_df.set_index('player_id')['own'].to_dict()

    # Precompute ownership buckets
    bucket_map = {
        pid: classify_ownership_bucket(own_map.get(pid, 0.0))
        for pid in proj_map.keys()
    }

    # 1. Setup the Problem
    prob = LpProblem("DFS_Lineup_Optimization", LpMaximize)

    # Decision variables: x_i = 1 if player i is selected
    player_vars = {
        pid: LpVariable(f"x_{pid}", lowBound=0, upBound=1, cat=LpBinary)
        for pid in proj_map.keys()
    }

    # 2. Objective: maximize total projection
    prob += lpSum([proj_map[pid] * player_vars[pid] for pid in proj_map.keys()]), "Total_Proj"

    # 3. Core Constraints
    # 3a. Roster size
    prob += lpSum([player_vars[pid] for pid in proj_map.keys()]) == template.roster_size, "Roster_Size"

    # 3b. Salary cap
    prob += lpSum([salary_map[pid] * player_vars[pid] for pid in proj_map.keys()]) <= template.salary_cap, "Salary_Cap"

    # 3c. Locked players must be selected
    for pid in locked_player_ids:
        if pid in player_vars:
            prob += player_vars[pid] == 1, f"Locked_{pid}"

    # 4. Ownership Bucket Constraints (soft via slack)
    template_buckets = template.to_dict()

    for bucket_name, required_count in template_buckets.items():
        # Baseball style: allow +/- bucket_slack
        lower_bound = max(required_count - bucket_slack, 0)
        upper_bound = required_count + bucket_slack

        prob += (
            lpSum(
                [
                    player_vars[pid]
                    for pid, bucket in bucket_map.items()
                    if bucket == bucket_name
                ]
            )
            >= lower_bound,
            f"{bucket_name}_lower",
        )
        prob += (
            lpSum(
                [
                    player_vars[pid]
                    for pid, bucket in bucket_map.items()
                    if bucket == bucket_name
                ]
            )
            <= upper_bound,
            f"{bucket_name}_upper",
        )

    # 5. Solve
    prob.solve(PULP_CBC_CMD(msg=False))

    if LpStatus[prob.status] != "Optimal":
        return None

    selected_ids = [
        pid for pid, var in player_vars.items() if var.varValue is not None and var.varValue > 0.5
    ]

    if len(selected_ids) != template.roster_size:
        return None

    return selected_ids


# --- MONTE CARLO SIMULATIONS + EXPOSURE / DIVERSITY ---

def run_monte_carlo_simulations(
    slate_df: pd.DataFrame, 
    template: 'LineupTemplate', 
    num_iterations: int,
    max_exposures: Dict[str, float],
    bucket_slack: int,
    locked_player_ids: List[str], 
    excluded_player_ids: List[str],
    min_lineup_diversity: int = 4
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Runs Monte Carlo simulations, collects optimal lineups, and applies 
    Max Exposure and Lineup Diversity constraints.
    Returns: (list of final lineups (dicts), dict of player exposures)
    """
    
    raw_optimal_lineups: List[Dict[str, Any]] = []
    sim_df = slate_df.copy()

    # --- Ensure numeric columns are valid floats ---
    try:
        sim_df['proj'] = pd.to_numeric(sim_df['proj'], errors='coerce')
        sim_df['salary'] = pd.to_numeric(sim_df['salary'], errors='coerce')
    except Exception as e:
        print(f"ERROR: conversion to float failed in run_monte_carlo_simulations: {e}")
        return [], {}

    sim_df.dropna(subset=['proj', 'salary'], inplace=True)

    if sim_df.empty:
        print("ERROR: sim_df empty after numeric cleaning in Monte Carlo.")
        return [], {}

    # 1. Pre-calculate per-player standard deviation as a % of projection
    sim_df['std_dev'] = sim_df['proj'].abs() * DEFAULT_STD_DEV_PCT
    sim_df.loc[sim_df['std_dev'] <= 0, 'std_dev'] = 0.1
    sim_df.loc[sim_df['proj'] <= 0, 'proj'] = 0.1

    # 2. Main simulation loop
    for i in range(num_iterations):
        loc_values = sim_df['proj'].to_numpy(dtype=float)
        scale_values = sim_df['std_dev'].to_numpy(dtype=float)

        # Guard against zero / negative std deviations
        scale_values = np.where(scale_values <= 0, 0.1, scale_values)

        sampled_values = np.random.normal(
            loc=loc_values,
            scale=scale_values,
            size=len(sim_df)
        )

        # IMPORTANT FIX: clip using np.clip (not .clip(lower=...))
        sampled_values = np.clip(sampled_values, a_min=0.1, a_max=None)

        # Use sampled projections for this iteration
        temp_df = sim_df.copy()
        temp_df['proj'] = sampled_values

        lineup_ids = optimize_single_lineup(
            slate_df=temp_df,
            template=template,
            bucket_slack=bucket_slack,
            locked_player_ids=locked_player_ids,
            excluded_player_ids=excluded_player_ids,
        )

        if lineup_ids:
            lineup_proj = temp_df.loc[
                temp_df['player_id'].isin(lineup_ids),
                'proj'
            ].sum()

            raw_optimal_lineups.append({
                'player_ids': lineup_ids,
                'proj_score': float(lineup_proj),
            })

    # 3. Post-simulation filtering: max exposure + lineup diversity
    if not raw_optimal_lineups:
        return [], {}

    # Sort by projected score, best first
    raw_optimal_lineups.sort(key=lambda x: x['proj_score'], reverse=True)

    final_lineups: List[Dict[str, Any]] = []
    player_counts: Dict[str, int] = {pid: 0 for pid in slate_df['player_id']}

    max_output_lineups = min(len(raw_optimal_lineups), 100)

    for lineup in raw_optimal_lineups:
        lineup_ids = lineup['player_ids']

        # A. Max exposure check (max_exposures values are 0–1 fractions)
        violates_exposure = False
        for pid in lineup_ids:
            current_total = len(final_lineups)
            current_count = player_counts.get(pid, 0)
            max_pct = max_exposures.get(pid, 1.0)

            if max_pct <= 0:
                violates_exposure = True
                break

            # exposure if we include this lineup next
            next_total = current_total + 1
            next_count = current_count + 1
            next_exposure = next_count / max(next_total, 1)

            if next_exposure - max_pct > 1e-9:
                violates_exposure = True
                break

        if violates_exposure:
            continue

        # B. Diversity check (limit shared players with existing lineups)
        is_diverse = True
        for existing in final_lineups:
            shared = len(set(lineup_ids) & set(existing['player_ids']))
            if shared > min_lineup_diversity:
                is_diverse = False
                break

        if not is_diverse:
            continue

        # Passed both gates; keep lineup
        final_lineups.append(lineup)
        for pid in lineup_ids:
            player_counts[pid] = player_counts.get(pid, 0) + 1

        if len(final_lineups) >= max_output_lineups:
            break

    total_lineups = len(final_lineups)
    if total_lineups == 0:
        return [], {}

    final_exposures = {
        pid: round((count / total_lineups) * 100, 1)
        for pid, count in player_counts.items()
        if count > 0
    }

    return final_lineups, final_exposures
