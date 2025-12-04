# builder.py

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import pandas as pd
import pulp

# These should match your thresholds in app.py
MEGA_CHALK_THR = 0.40  # >= 40% owned
CHALK_THR      = 0.30  # 30–39%
PUNT_THR       = 0.10  # < 10%

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8  # DK NBA-style, change if needed


# ---------------- Ownership buckets -----------------

def ownership_bucket(own: float) -> str:
    """Return ownership bucket name for a given ownership fraction (0-1)."""
    if own >= MEGA_CHALK_THR:
        return "mega"
    elif own >= CHALK_THR:
        return "chalk"
    elif own >= PUNT_THR:
        return "mid"
    else:
        return "punt"


@dataclass
class StructureTemplate:
    contest_label: str
    roster_size: int
    salary_cap: int
    target_mega: float
    target_chalk: float
    target_mid: float
    target_punt: float

    def bucket_ranges(self, slack: int = 1) -> Dict[str, Tuple[int, int]]:
        """
        Turn float targets into min/max integer ranges for each bucket.
        slack = how much wiggle room per bucket.
        """
        def clip_pair(x: float):
            base = round(x)
            return max(0, base - slack), max(0, base + slack)

        mega_min, mega_max = clip_pair(self.target_mega)
        chalk_min, chalk_max = clip_pair(self.target_chalk)
        mid_min, mid_max     = clip_pair(self.target_mid)
        punt_min, punt_max   = clip_pair(self.target_punt)

        return {
            "mega":  (mega_min, mega_max),
            "chalk": (chalk_min, chalk_max),
            "mid":   (mid_min, mid_max),
            "punt":  (punt_min, punt_max),
        }


# ---------------- Contest params → template -----------------

def build_template_from_params(
    contest_type: str,
    field_size: int,
    pct_to_first: float,
    roster_size: int = DEFAULT_ROSTER_SIZE,
    salary_cap: int = DEFAULT_SALARY_CAP,
) -> StructureTemplate:
    """
    Heuristic mapping from contest_type + field_size + % to first
    → target bucket counts per lineup.

    pct_to_first is the % of the total prize pool that goes to 1st (e.g. 25 for 25%).
    """
    ct = contest_type.upper()
    top_heavy = pct_to_first >= 20
    large_field = field_size >= 5000
    small_field = field_size <= 1000

    # Default “balanced GPP” starting point
    target_mega = 2.0
    target_chalk = 2.5
    target_mid = 2.5
    target_punt = 1.0
    label = f"{ct}_GENERIC"

    if ct == "CASH":
        label = "CASH"
        target_mega = 3.5
        target_chalk = 3.0
        target_mid = 1.0
        target_punt = 0.5

    elif ct == "SE":
        if small_field and not top_heavy:
            # small SE, flatter payouts -> safer
            label = "SE_SMALL_FLAT"
            target_mega = 2.5
            target_chalk = 2.5
            target_mid = 2.0
            target_punt = 1.0
        elif small_field and top_heavy:
            label = "SE_SMALL_TOPHEAVY"
            target_mega = 2.0
            target_chalk = 2.0
            target_mid = 2.0
            target_punt = 2.0
        elif large_field and top_heavy:
            label = "SE_BIG_TOPHEAVY"
            target_mega = 1.5
            target_chalk = 2.0
            target_mid = 2.0
            target_punt = 2.5
        else:
            label = "SE_GENERIC"
            target_mega = 2.0
            target_chalk = 2.5
            target_mid = 2.0
            target_punt = 1.5

    elif ct == "3MAX":
        if top_heavy or large_field:
            label = "3MAX_AGGRO"
            target_mega = 1.5
            target_chalk = 2.0
            target_mid = 2.5
            target_punt = 2.0
        else:
            label = "3MAX_BALANCED"
            target_mega = 2.0
            target_chalk = 2.5
            target_mid = 2.0
            target_punt = 1.5

    elif ct == "MME":
        if large_field and top_heavy:
            label = "MME_LARGE_TOPHEAVY"
            target_mega = 1.5
            target_chalk = 2.0
            target_mid = 2.5
            target_punt = 2.0
        else:
            label = "MME_SMALLER"
            target_mega = 1.5
            target_chalk = 2.5
            target_mid = 2.0
            target_punt = 2.0

    return StructureTemplate(
        contest_label=label,
        roster_size=roster_size,
        salary_cap=salary_cap,
        target_mega=target_mega,
        target_chalk=target_chalk,
        target_mid=target_mid,
        target_punt=target_punt,
    )


# ---------------- Optional: standalone slate loader -----------------

def load_slate_players(path: str) -> pd.DataFrame:
    """
    Standalone loader for slate CSV (if using this module by itself).
    Same expectations as the app.
    """
    df = pd.read_csv(path)

    rename_map = {}
    if "Name" in df.columns:
        rename_map["Name"] = "name"
    if "Salary" in df.columns:
        rename_map["Salary"] = "salary"
    if "Projection" in df.columns:
        rename_map["Projection"] = "proj"
    if "Proj" in df.columns:
        rename_map["Proj"] = "proj"
    if "Own" in df.columns:
        rename_map["Own"] = "own_proj"
    if "Pos" in df.columns:
        rename_map["Pos"] = "positions"
    if "Positions" in df.columns:
        rename_map["Positions"] = "positions"

    df = df.rename(columns=rename_map)

    required = ["player_id", "name", "salary", "proj", "own_proj"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Slate file missing required columns: {missing}")

    df["bucket"] = df["own_proj"].apply(ownership_bucket)
    return df


# ---------------- Optimizer: build a single lineup -----------------

def build_optimal_lineup(
    slate_df: pd.DataFrame,
    template: StructureTemplate,
    bucket_slack: int = 1,
    avoid_player_ids: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Build one optimal lineup for the given contest template.

    - Maximizes proj
    - Respects salary cap
    - Uses exactly roster_size players
    - Enforces bucket count ranges (mega/chalk/mid/punt) with some slack
    - Optionally avoids specific player_ids
    """
    df = slate_df.copy().reset_index(drop=True)

    if "bucket" not in df.columns:
        df["bucket"] = df["own_proj"].apply(ownership_bucket)

    if avoid_player_ids:
        df = df[~df["player_id"].isin(avoid_player_ids)].reset_index(drop=True)

    n = df.shape[0]
    if n == 0:
        return None

    # Define problem
    prob = pulp.LpProblem("DFS_Lineup", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n), lowBound=0, upBound=1, cat="Binary")

    # Objective: maximize projection
    prob += pulp.lpSum(df.loc[i, "proj"] * x[i] for i in range(n)), "Total_Projection"

    # Roster size
    prob += pulp.lpSum(x[i] for i in range(n)) == template.roster_size, "RosterSize"

    # Salary
    prob += pulp.lpSum(df.loc[i, "salary"] * x[i] for i in range(n)) <= template.salary_cap, "SalaryCap"

    # Bucket constraints
    bucket_ranges = template.bucket_ranges(slack=bucket_slack)
    for bucket_name, (bmin, bmax) in bucket_ranges.items():
        idxs = [i for i in range(n) if df.loc[i, "bucket"] == bucket_name]
        if not idxs:
            continue
        prob += pulp.lpSum(x[i] for i in idxs) >= bmin, f"{bucket_name}_min"
        prob += pulp.lpSum(x[i] for i in idxs) <= bmax, f"{bucket_name}_max"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        print("No optimal lineup found (status:", pulp.LpStatus[prob.status], ")")
        return None

    selected = [df.loc[i] for i in range(n) if x[i].varValue == 1]
    if not selected:
        return None

    lineup = pd.DataFrame(selected).reset_index(drop=True)
    lineup["bucket"] = lineup["own_proj"].apply(ownership_bucket)
    return lineup
