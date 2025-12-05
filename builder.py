# builder.py

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import pulp

# --- GLOBAL CONFIGURATION ---
MEGA_CHALK_THR = 0.40    # >= 40% owned
CHALK_THR      = 0.30    # 30–39%
PUNT_THR       = 0.10    # < 10%

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8  # Classic DK format (NBA)

# --- NBA CLASSIC ROSTER SLOTS ---
# Defines the minimum number of players required for each basic position group.
ROSTER_REQUIREMENTS = {
    'PG': 1,
    'SG': 1,
    'SF': 1,
    'PF': 1,
    'C': 1,
    'G': 1,  # Guard (PG or SG)
    'F': 1,  # Forward (SF or PF)
    # UTIL is covered by the total roster size constraint
}

MIN_GAMES_REQUIRED = 2 # DK rule: Lineups must include players from at least 2 different games.


# ---------------- Ownership Buckets ---------------- #

def ownership_bucket(own: float) -> str:
    """Return ownership bucket name based on ownership projection."""
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
    """Defines the target ownership structure and rules for a contest type."""
    contest_label: str
    roster_size: int
    salary_cap: int
    target_mega: float
    target_chalk: float
    target_mid: float
    target_punt: float
    roster_slots: Dict[str, int] = field(default_factory=lambda: ROSTER_REQUIREMENTS)
    min_games: int = MIN_GAMES_REQUIRED

    def bucket_ranges(self, slack: int = 1) -> Dict[str, Tuple[int, int]]:
        """Convert float targets → integer min/max with slack."""
        def clip_pair(x: float):
            # Target is a float count (e.g., 2.5 players)
            base = round(x)
            # Slack allows the optimizer flexibility (e.g., 2.5 +/- 1 = [2, 3])
            return max(0, base - slack), max(0, base + slack)

        return {
            "mega":  clip_pair(self.target_mega),
            "chalk": clip_pair(self.target_chalk),
            "mid":   clip_pair(self.target_mid),
            "punt":  clip_pair(self.target_punt),
        }

# ---------------- Template Generator ---------------- #
# (Heuristic logic remains the same - generating ownership targets based on contest type)

def build_template_from_params(
    contest_type: str,
    field_size: int,
    pct_to_first: float,
    roster_size: int = DEFAULT_ROSTER_SIZE,
    salary_cap: int = DEFAULT_SALARY_CAP,
    roster_slots: Dict[str, int] = ROSTER_REQUIREMENTS,
    min_games: int = MIN_GAMES_REQUIRED,
) -> StructureTemplate:
    """Heuristic mapping: contest traits → ideal ownership structure."""
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
        # ... (rest of SE and other contest type logic here) ...
        # (Using the generic one for brevity, but the logic in the previous response applies)
        else:
            label = "SE_GENERIC"
            target_mega = 2.0
            target_chalk = 2.5
            target_mid = 2.0
            target_punt = 1.5


    return StructureTemplate(
        contest_label=label,
        roster_size=roster_size,
        salary_cap=salary_cap,
        target_mega=target_mega,
        target_chalk=target_chalk,
        target_mid=target_mid,
        target_punt=target_punt,
        roster_slots=roster_slots,
        min_games=min_games,
    )


# ---------------- Classic Lineup Optimizer (ENHANCED) ---------------- #

def build_optimal_lineup(
    slate_df: pd.DataFrame,
    template: StructureTemplate,
    bucket_slack: int = 1,
    avoid_player_ids: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Classic DK (e.g., NBA 8-man): Maximize projection under cap and constraints.
    Includes Positional, Ownership Bucket, and Game Diversity constraints.
    """
    df = slate_df.copy().reset_index(drop=True)

    if "bucket" not in df.columns:
        df["bucket"] = df["own_proj"].apply(ownership_bucket)

    if avoid_player_ids:
        df = df[~df["player_id"].isin(avoid_player_ids)].reset_index(drop=True)

    if df.empty:
        return None

    player_ids = df['player_id'].tolist()
    
    # 1. Initialize the Problem and Variables
    prob = pulp.LpProblem("DFS_Lineup_Classic", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", player_ids, lowBound=0, upBound=1, cat="Binary")
    
    # Objective: maximize projection
    prob += pulp.lpSum(df.loc[df['player_id'] == i, "proj"].iloc[0] * x[i] for i in player_ids), "Maximize_Total_Points"

    # 2. General Constraints
    prob += pulp.lpSum(x[i] for i in player_ids) == template.roster_size, "Exactly_8_Players"
    prob += pulp.lpSum(df.loc[df['player_id'] == i, "salary"].iloc[0] * x[i] for i in player_ids) <= template.salary_cap, "Salary_Cap"

    # 3. Ownership Bucket Ranges (Leverage Constraint)
    bucket_ranges = template.bucket_ranges(slack=bucket_slack)
    for bucket_name, (bmin, bmax) in bucket_ranges.items():
        ids = df[df['bucket'] == bucket_name]['player_id'].tolist()
        if ids:
            prob += pulp.lpSum(x[i] for i in ids) >= bmin, f"Min_{bucket_name}"
            prob += pulp.lpSum(x[i] for i in ids) <= bmax, f"Max_{bucket_name}"

    # 4. Positional Constraints
    def is_eligible(player_position_string, slot_position):
        """Checks if a player is eligible for a given slot position."""
        return slot_position in player_position_string.split('/')

    for pos, required in template.roster_slots.items():
        # Determine the base positions that satisfy this slot (e.g., 'G' needs 'PG' or 'SG')
        if pos == 'G':
            eligible_pos = ['PG', 'SG']
        elif pos == 'F':
            eligible_pos = ['SF', 'PF']
        else:
            eligible_pos = [pos] # Single position
            
        # Get IDs of players eligible for this slot
        eligible_ids = df[
            df['positions'].apply(lambda p: any(is_eligible(p, ep) for ep in eligible_pos))
        ]['player_id'].tolist()
        
        if eligible_ids:
            # Sum of selected players eligible for the slot must meet the minimum
            # 
            prob += (
                pulp.lpSum(x[i] for i in eligible_ids) >= required,
                f"Min_{required}_Players_for_{pos}"
            )


    # 5. Game Diversity Constraint (Requires players from >= N games)
    if 'GameID' in df.columns and template.min_games > 1:
        unique_game_ids = df['GameID'].unique().tolist()
        
        # Indicator variables: 1 if a player from that game is selected
        game_selected_vars = pulp.LpVariable.dicts(
            "GameSelected", unique_game_ids, 0, 1, pulp.LpBinary
        )

        # Link player selection to game selection (Big M Constraint)
        for game_id in unique_game_ids:
            players_in_game = df[df['GameID'] == game_id]['player_id'].tolist()
            
            # Sum(Players in Game j) <= M * GameSelected_j. M = roster_size (8)
            prob += (
                pulp.lpSum(x[i] for i in players_in_game) <= template.roster_size * game_selected_vars[game_id],
                f"Link_Players_to_Game_{game_id}"
            )

        # Enforce the Minimum Diversity Rule
        prob += (
            pulp.lpSum(game_selected_vars[game_id] for game_id in unique_game_ids) >= template.min_games,
            "Minimum_Game_Diversity"
        )

    # 6. Solve and Extract
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    selected_ids = [i for i in player_ids if x[i].varValue == 1]
    if not selected_ids:
        return None

    lineup = df[df['player_id'].isin(selected_ids)].copy()
    lineup["bucket"] = lineup["own_proj"].apply(ownership_bucket)
    return lineup.reset_index(drop=True)


# ---------------- Showdown Slate Expansion and Optimization ---------------- #

# (The Showdown functions are omitted for brevity, but would be included here 
# if you needed that mode, using the logic from the previous builder.py response.)
