# builder.py

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import pandas as pd
import pulp

# Ownership buckets
MEGA_CHALK_THR = 0.40   # >= 40% owned
CHALK_THR      = 0.30   # 30–39%
PUNT_THR       = 0.10   # < 10%

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8  # Classic DK format


# ---------------- Ownership Buckets ---------------- #

def ownership_bucket(own: float) -> str:
    """Return ownership bucket name."""
    if own >= MEGA_CHALK_THR:
        return "mega"
    elif own >= CHALK_THR:
        return "chalk"
    elif own >= PUNT_THR:
        return "mid"
    else:
        return "punt"


# ---------------- Contest Template ---------------- #

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
        """Convert float targets → integer min/max with slack."""
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


# ---------------- Template Generator ---------------- #

def build_template_from_params(
    contest_type: str,
    field_size: int,
    pct_to_first: float,
    roster_size: int = DEFAULT_ROSTER_SIZE,
    salary_cap: int = DEFAULT_SALARY_CAP,
) -> StructureTemplate:
    """
    Heuristic mapping: contest traits → ideal ownership structure.
    """
    ct = contest_type.upper()
    top_heavy = pct_to_first >= 20
    large_field = field_size >= 5000
    small_field = field_size <= 1000

    # Default "balanced GPP"
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


# ---------------- Classic Lineup Optimizer ---------------- #

def build_optimal_lineup(
    slate_df: pd.DataFrame,
    template: StructureTemplate,
    bucket_slack: int = 1,
    avoid_player_ids: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Classic DK: choose `template.roster_size` players under cap,
    maximize projection, respecting ownership bucket counts.
    """
    df = slate_df.copy().reset_index(drop=True)

    if "bucket" not in df.columns:
        df["bucket"] = df["own_proj"].apply(ownership_bucket)

    if avoid_player_ids:
        df = df[~df["player_id"].isin(avoid_player_ids)].reset_index(drop=True)

    n = df.shape[0]
    if n == 0:
        return None

    prob = pulp.LpProblem("DFS_Lineup_Classic", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n), lowBound=0, upBound=1, cat="Binary")

    # Objective: maximize projection
    prob += pulp.lpSum(df.loc[i, "proj"] * x[i] for i in range(n))

    # Roster size constraint
    prob += pulp.lpSum(x[i] for i in range(n)) == template.roster_size

    # Salary constraint
    prob += pulp.lpSum(df.loc[i, "salary"] * x[i] for i in range(n)) <= template.salary_cap

    # Ownership bucket ranges
    bucket_ranges = template.bucket_ranges(slack=bucket_slack)
    for bucket_name, (bmin, bmax) in bucket_ranges.items():
        idxs = [i for i in range(n) if df.loc[i, "bucket"] == bucket_name]
        if idxs:
            prob += pulp.lpSum(x[i] for i in idxs) >= bmin
            prob += pulp.lpSum(x[i] for i in idxs) <= bmax

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    selected = [df.loc[i] for i in range(n) if x[i].varValue == 1]
    if not selected:
        return None

    lineup = pd.DataFrame(selected)
    lineup["bucket"] = lineup["own_proj"].apply(ownership_bucket)
    return lineup


# ---------------- Showdown Slate Expansion ---------------- #

def expand_showdown_slate(slate_df: pd.DataFrame) -> pd.DataFrame:
    """
    For showdown: create CPT (1.5x salary/proj) and FLEX versions of each player.
    """
    base = slate_df.copy()
    base["player_id"] = base["player_id"].astype(str)
    base["base_id"] = base["player_id"]
    base["role"] = "FLEX"

    cpt = base.copy()
    cpt["role"] = "CPT"
    cpt["salary"] = (cpt["salary"] * 1.5).round().astype(int)
    cpt["proj"] = cpt["proj"] * 1.5
    cpt["player_id"] = cpt["player_id"] + "_CPT"

    df = pd.concat([base, cpt], ignore_index=True)

    if "bucket" not in df.columns:
        df["bucket"] = df["own_proj"].apply(ownership_bucket)

    return df


# ---------------- Showdown Lineup Optimizer ---------------- #

def build_optimal_lineup_showdown(
    slate_df: pd.DataFrame,
    template: StructureTemplate,
    bucket_slack: int = 1,
    avoid_player_ids: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Showdown: pick exactly 6 players (1 CPT, 5 FLEX) under $50k.

    `template.roster_size` should be 6 and `template.salary_cap` 50000.
    """
    df = expand_showdown_slate(slate_df).reset_index(drop=True)

    if avoid_player_ids:
        df = df[~df["player_id"].isin(avoid_player_ids)].reset_index(drop=True)

    n = df.shape[0]
    if n == 0:
        return None

    prob = pulp.LpProblem("DFS_Lineup_Showdown", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(n), lowBound=0, upBound=1, cat="Binary")

    # Objective
    prob += pulp.lpSum(df.loc[i, "proj"] * x[i] for i in range(n))

    # Exactly 6 players
    prob += pulp.lpSum(x[i] for i in range(n)) == template.roster_size

    # Salary cap
    prob += pulp.lpSum(df.loc[i, "salary"] * x[i] for i in range(n)) <= template.salary_cap

    # Exactly 1 CPT
    cpt_idxs = [i for i in range(n) if df.loc[i, "role"] == "CPT"]
    prob += pulp.lpSum(x[i] for i in cpt_idxs) == 1

    # Prevent CPT+FLEX duplication of same player
    for base_id, idxs in df.groupby("base_id").groups.items():
        prob += pulp.lpSum(x[i] for i in idxs) <= 1

    # Ownership bucket constraints
    bucket_ranges = template.bucket_ranges(slack=bucket_slack)
    for bucket_name, (bmin, bmax) in bucket_ranges.items():
        idxs = [i for i in range(n) if df.loc[i, "bucket"] == bucket_name]
        if idxs:
            prob += pulp.lpSum(x[i] for i in idxs) >= bmin
            prob += pulp.lpSum(x[i] for i in idxs) <= bmax

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    selected = [df.loc[i] for i in range(n) if x[i].varValue == 1]
    if not selected:
        return None

    lineup = pd.DataFrame(selected)
    lineup["bucket"] = lineup["own_proj"].apply(ownership_bucket)
    return lineup
