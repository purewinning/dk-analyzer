from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import pandas as pd
import pulp

# Ownership bucket thresholds — must match app.py!
# Mega Chalk: Highly owned, high floor/ceiling
MEGA_CHALK_THR = 0.40    # >= 40% owned
# Chalk: Solid play, moderately high ownership
CHALK_THR      = 0.30    # 30–39%
# Mid-Range: Moderate ownership, often the leverage zone
PUNT_THR       = 0.10    # < 10%

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8  # Classic DK format


# ---------------- Ownership Buckets ---------------- #

def ownership_bucket(own: float) -> str:
    """Return ownership bucket name based on the defined thresholds."""
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
    """Defines the target composition of a roster based on contest type."""
    contest_label: str
    roster_size: int
    salary_cap: int
    # Target number of players from each bucket
    target_mega: float
    target_chalk: float
    target_mid: float
    target_punt: float

    def bucket_ranges(self, slack: int = 1) -> Dict[str, Tuple[int, int]]:
        """
        Convert float targets into integer min/max counts for the solver, 
        using a specified slack (tolerance).
        """
        def clip_pair(x: float) -> Tuple[int, int]:
            base = round(x)
            # Ensure min is never negative
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
    Heuristic mapping: contest traits → ideal ownership structure targets.
    This aims to provide leverage based on contest size and payout structure.
    """
    ct = contest_type.upper()
    top_heavy = pct_to_first >= 20.0  # e.g., Winner takes 20% or more
    large_field = field_size >= 5000
    small_field = field_size <= 1000

    # Default "balanced GPP" (The original starting point)
    target_mega = 2.0
    target_chalk = 2.5
    target_mid = 2.5
    target_punt = 1.0
    label = f"{ct}_GENERIC"

    if ct == "CASH":
        # Cash games prioritize high floor/projection over differentiation.
        label = "CASH"
        target_mega = 3.5  # Heavy on the safest, highly owned plays
        target_chalk = 3.0
        target_mid = 1.0
        target_punt = 0.5  # Minimal risk
    
    elif ct == "SE": # Single Entry
        if small_field and not top_heavy:
            # Small, flat SE: Can lean slightly towards safer plays
            label = "SE_SMALL_FLAT"
            target_mega = 2.5
            target_chalk = 2.5
            target_mid = 2.0
            target_punt = 1.0
        elif small_field and top_heavy:
            # Small, top-heavy SE: Need more differentiation for first place
            label = "SE_SMALL_TOPHEAVY"
            target_mega = 2.0
            target_chalk = 2.0
            target_mid = 3.0
            target_punt = 1.0
        else: # Large Field SE (e.g., Mini Max)
            # Must differentiate more heavily than small SE
            label = "SE_LARGE_GPP"
            target_mega = 1.5
            target_chalk = 2.0
            target_mid = 3.5
            target_punt = 1.0

    elif ct == "MME": # Mass Multi-Entry (Tournaments)
        if large_field:
            # Maximum differentiation/leverage for huge contests
            label = "MME_LARGE"
            target_mega = 1.0
            target_chalk = 2.0
            target_mid = 3.5
            target_punt = 1.5
        else:
            # Standard MME differentiation
            label = "MME_GENERIC"
            target_mega = 1.5
            target_chalk = 2.0
            target_mid = 3.0
            target_punt = 1.5
    
    # Ensure targets sum up to Roster Size (or close to it)
    total_target = target_mega + target_chalk + target_mid + target_punt
    if total_target != roster_size:
        # Simple normalization to ensure sum is correct
        factor = roster_size / total_target
        target_mega *= factor
        target_chalk *= factor
        target_mid *= factor
        target_punt *= factor

    return StructureTemplate(
        contest_label=label,
        roster_size=roster_size,
        salary_cap=salary_cap,
        target_mega=target_mega,
        target_chalk=target_chalk,
        target_mid=target_mid,
        target_punt=target_punt,
    )


# ---------------- Optimization Function (LP Solver) ---------------- #

def optimize_lineup(
    players_df: pd.DataFrame, 
    template: StructureTemplate,
) -> Optional[List[str]]:
    """
    Uses Linear Programming (PuLP) to find the highest projected lineup 
    that satisfies salary, roster, and ownership bucket constraints.
    
    Assumes players_df has columns: 'Name', 'Position', 'Salary', 'Proj', 'Ownership', 'Bucket'.
    """
    
    # --- 1. Setup Data and Problem ---
    prob = pulp.LpProblem("DFS Lineup Optimizer", pulp.LpMaximize)
    
    # Decision Variables: A binary variable for each player (1 if in lineup, 0 otherwise)
    player_vars = pulp.LpVariable.dicts(
        "Player", players_df['Name'], 0, 1, pulp.LpBinary
    )

    # --- 2. Objective Function: Maximize Projected Points ---
    prob += pulp.lpSum(
        players_df.loc[i, 'Proj'] * player_vars[name] 
        for i, name in enumerate(players_df['Name'])
    ), "Total Projected Points"

    # --- 3. Constraints ---
    
    # a) Salary Cap Constraint: Total salary must be <= CAP
    prob += pulp.lpSum(
        players_df.loc[i, 'Salary'] * player_vars[name]
        for i, name in enumerate(players_df['Name'])
    ) <= template.salary_cap, "Salary Cap"

    # b) Roster Size Constraint: Select exactly RosterSize players
    prob += pulp.lpSum(
        player_vars[name] for name in players_df['Name']
    ) == template.roster_size, "Roster Size"

    # c) Position Constraints (Example: assuming 1QB, 2RB, 3WR, 1TE, 1DST for 8 spots)
    # NOTE: In a real DK setup, this is more complex (e.g., RB/WR/TE FLEX)
    position_counts = players_df.groupby('Position').size()
    
    # Simple constraints for illustration
    if 'QB' in players_df['Position'].values:
        prob += pulp.lpSum(
            player_vars[name] for i, name in enumerate(players_df['Name']) 
            if players_df.loc[i, 'Position'] == 'QB'
        ) == 1, "Select 1 QB"
    
    # d) Ownership Bucket Constraints
    bucket_ranges = template.bucket_ranges(slack=1)
    
    for bucket, (min_c, max_c) in bucket_ranges.items():
        # Sum of players selected within this ownership bucket
        sum_bucket = pulp.lpSum(
            player_vars[name] for i, name in enumerate(players_df['Name']) 
            if players_df.loc[i, 'Bucket'] == bucket
        )
        
        # Min constraint
        prob += sum_bucket >= min_c, f"Min {bucket.capitalize()} Players"
        # Max constraint
        prob += sum_bucket <= max_c, f"Max {bucket.capitalize()} Players"

    # --- 4. Solve and Extract Result ---
    prob.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 suppresses solver output
    
    if prob.status == pulp.LpStatusOptimal:
        selected_players = [
            name for name, var in player_vars.items() 
            if var.varValue > 0.5
        ]
        return selected_players
    else:
        # print(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
        return None

# ---------------- Example Usage ---------------- #

if __name__ == '__main__':
    # 1. Create Mock Player Data
    data = {
        'Name': [f'Player_{i}' for i in range(1, 15)],
        'Position': ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'DST'] + ['QB', 'RB', 'WR', 'WR', 'TE', 'DST'],
        'Salary': [6500, 7800, 4500, 8100, 3200, 5000, 4000, 3000, 
                   5800, 7000, 6000, 5500, 4000, 3500],
        'Proj': [25.5, 20.1, 15.0, 28.0, 10.5, 12.0, 9.5, 8.0,
                 22.0, 18.5, 14.5, 13.0, 11.0, 7.5],
        'Ownership': [0.45, 0.35, 0.15, 0.40, 0.05, 0.31, 0.10, 0.08,
                      0.25, 0.12, 0.09, 0.20, 0.42, 0.33],
    }
    players_df = pd.DataFrame(data)
    
    # 2. Add Ownership Bucket
    players_df['Bucket'] = players_df['Ownership'].apply(ownership_bucket)
    
    # 3. Generate Template for a Large GPP
    # Example: MME, Field Size 50000, 25% to first
    template = build_template_from_params("MME", 50000, 25.0, roster_size=8)
    
    print(f"--- Optimization for {template.contest_label} (Roster Size: {template.roster_size}, Cap: ${template.salary_cap}) ---")
    print("Target Ranges (Min, Max) with Slack=1:")
    print(template.bucket_ranges(slack=1))
    
    # 4. Run Optimization
    optimal_lineup = optimize_lineup(players_df, template)
    
    if optimal_lineup:
        lineup_df = players_df[players_df['Name'].isin(optimal_lineup)]
        print("\n--- Optimal Lineup Found ---")
        print(lineup_df[['Name', 'Position', 'Salary', 'Proj', 'Ownership', 'Bucket']])
        print(f"\nTotal Salary: ${lineup_df['Salary'].sum()}")
        print(f"Total Projection: {lineup_df['Proj'].sum():.2f}")
        print("Bucket Counts:")
        print(lineup_df['Bucket'].value_counts())
    else:
        print("\nCould not find a feasible lineup meeting all constraints.")
