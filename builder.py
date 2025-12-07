\import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# --- CONSTANTS ---

DEFAULT_SALARY_CAP = 50000
DEFAULT_ROSTER_SIZE = 8

# Ownership bucket thresholds (in percent) – for visual buckets
PUNT_THR = 10.0       # < 10% = punt / contrarian
CHALK_THR = 30.0      # 10–30 mid, 30–40 chalk
MEGA_CHALK_THR = 40.0 # > 40 mega-chalk


# --- TEMPLATE OBJECT --------------------------------------------------------


@dataclass
class LineupTemplate:
    contest_type: str
    field_size: int
    pct_to_first: float
    roster_size: int = DEFAULT_ROSTER_SIZE
    salary_cap: int = DEFAULT_SALARY_CAP
    min_games: int = 2


def build_template_from_params(
    contest_type: str,
    field_size: int,
    pct_to_first: float,
    roster_size: int = DEFAULT_ROSTER_SIZE,
    salary_cap: int = DEFAULT_SALARY_CAP,
    min_games: int = 2,
) -> LineupTemplate:
    """
    Simple wrapper used by app.py to construct a template object.

    The template itself is intentionally lightweight; almost all of the
    strategy / ownership logic now lives in app.py where we can expose it
    cleanly to the user.
    """
    return LineupTemplate(
        contest_type=contest_type,
        field_size=field_size,
        pct_to_first=pct_to_first,
        roster_size=roster_size,
        salary_cap=salary_cap,
        min_games=min_games,
    )


# --- OWNERSHIP BUCKETING ----------------------------------------------------


def ownership_bucket(own: float) -> str:
    """Map projected ownership into a coarse bucket (for display)."""
    if pd.isna(own):
        return "mid"
    if own >= MEGA_CHALK_THR:
        return "mega"
    if own >= CHALK_THR:
        return "chalk"
    if own >= PUNT_THR:
        return "mid"
    return "punt"


# --- CORRELATION & STACKING HELPERS -----------------------------------------


def build_game_environments(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyze each game to identify high-upside environments.
    Returns dict keyed by GameID with metadata about the game.
    """
    game_envs = {}
    
    for game_id in df["GameID"].unique():
        game_players = df[df["GameID"] == game_id].copy()
        
        # Get the two teams
        teams = game_players["Team"].unique()
        if len(teams) != 2:
            continue
            
        team_a, team_b = teams[0], teams[1]
        
        # Calculate game totals and spreads
        team_a_proj = game_players[game_players["Team"] == team_a]["proj"].sum()
        team_b_proj = game_players[game_players["Team"] == team_b]["proj"].sum()
        
        total_proj = team_a_proj + team_b_proj
        spread = abs(team_a_proj - team_b_proj)
        
        # High ceiling games are close games with high totals
        # Blowouts reduce correlation benefit
        game_quality_score = total_proj * (1 - (spread / total_proj) * 0.3)
        
        # Calculate average ceiling of top players
        top_players = game_players.nlargest(6, "ceiling")
        avg_ceiling = top_players["ceiling"].mean()
        
        game_envs[game_id] = {
            "teams": [team_a, team_b],
            "total_proj": total_proj,
            "spread": spread,
            "quality_score": game_quality_score,
            "avg_ceiling": avg_ceiling,
            "is_high_upside": total_proj > game_players["proj"].mean() * 16,  # Above avg
        }
    
    return game_envs


def build_team_stacks(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identify viable team stacks (primary + secondary players).
    Returns dict keyed by Team with list of stack combinations.
    """
    team_stacks = {}
    
    for team in df["Team"].unique():
        team_players = df[df["Team"] == team].copy()
        
        if len(team_players) < 2:
            continue
        
        # Identify stars (top 30% salary on team)
        salary_threshold = team_players["salary"].quantile(0.7)
        stars = team_players[team_players["salary"] >= salary_threshold]
        
        # Identify role players (good value, lower ownership)
        role_players = team_players[
            (team_players["salary"] < salary_threshold) &
            (team_players["value"] >= team_players["value"].quantile(0.4))
        ]
        
        stacks = []
        for _, star in stars.iterrows():
            for _, role in role_players.iterrows():
                if star["player_id"] == role["player_id"]:
                    continue
                
                # Calculate stack quality
                combined_own = star["own_proj"] + role["own_proj"]
                combined_ceiling = star["ceiling"] + role["ceiling"]
                combined_proj = star["proj"] + role["proj"]
                
                # Lower combined ownership = better differentiation
                ownership_diff = 100 - combined_own
                
                stack_score = (
                    combined_ceiling * 0.4 +
                    combined_proj * 0.3 +
                    ownership_diff * 0.3
                )
                
                stacks.append({
                    "primary": star["player_id"],
                    "secondary": role["player_id"],
                    "combined_own": combined_own,
                    "combined_ceiling": combined_ceiling,
                    "stack_score": stack_score,
                })
        
        team_stacks[team] = sorted(stacks, key=lambda x: x["stack_score"], reverse=True)[:10]
    
    return team_stacks


def identify_bringback_candidates(
    df: pd.DataFrame,
    primary_team: str,
    game_id: str,
    max_candidates: int = 5
) -> List[str]:
    """
    Find optimal bring-back players from the opposing team.
    Best bring-backs: high ceiling, lower ownership, positional correlation.
    """
    # Get opposing team
    game_teams = df[df["GameID"] == game_id]["Team"].unique()
    opposing_team = [t for t in game_teams if t != primary_team]
    
    if not opposing_team:
        return []
    
    opposing_team = opposing_team[0]
    candidates = df[df["Team"] == opposing_team].copy()
    
    if candidates.empty:
        return []
    
    # Score bring-back candidates
    # Prefer: high ceiling, lower ownership, good value
    candidates["bringback_score"] = (
        candidates["ceiling"] * 0.4 +
        (100 - candidates["own_proj"]) * 0.3 +
        candidates["value"] * 10 * 0.3
    )
    
    top_candidates = candidates.nlargest(max_candidates, "bringback_score")
    return top_candidates["player_id"].tolist()


def calculate_lineup_correlation_score(
    lineup_df: pd.DataFrame,
    game_envs: Dict[str, Dict[str, Any]]
) -> float:
    """
    Score a lineup based on correlation strength.
    Higher score = better correlation/stacking.
    """
    if lineup_df.empty:
        return 0.0
    
    score = 0.0
    
    # Count players per game
    game_counts = lineup_df.groupby("GameID").size()
    
    for game_id, count in game_counts.items():
        if count >= 2:
            # Bonus for game stacks (multiple players from same game)
            game_info = game_envs.get(game_id, {})
            quality = game_info.get("quality_score", 0)
            
            # Stack bonus scales with number of players and game quality
            stack_bonus = (count - 1) * quality * 0.1
            score += stack_bonus
    
    # Count players per team
    team_counts = lineup_df.groupby("Team").size()
    
    for team, count in team_counts.items():
        if count >= 2:
            # Bonus for team stacks
            # Diminishing returns after 3 players from same team
            team_bonus = min(count - 1, 3) * 15
            score += team_bonus
    
    # Penalty for too many different games (reduces correlation)
    num_games = len(game_counts)
    if num_games > 4:
        score -= (num_games - 4) * 10
    
    return score


# --- LINEUP GENERATION ------------------------------------------------------


def _prepare_player_pool(
    slate_df: pd.DataFrame,
    locked_player_ids: List[str],
    excluded_player_ids: List[str],
) -> pd.DataFrame:
    """
    Return a clean player pool.

    Assumes slate_df already has columns:
        - player_id (str)
        - salary (int)
        - proj (float)
        - own_proj (float)  [optional, but handy]
        - GameID (str)
        - Team (str)
    """
    df = slate_df.copy()

    # Ensure needed columns exist
    required_cols = ["player_id", "salary", "proj", "GameID", "Team"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Slate is missing required column: {col}")

    # Drop excluded
    if excluded_player_ids:
        df = df[~df["player_id"].isin(excluded_player_ids)]

    # Make types safe
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    df["proj"] = pd.to_numeric(df["proj"], errors="coerce").fillna(0.0)

    # Remove totally unplayable rows
    df = df[df["salary"] > 0]
    df = df[df["proj"] > 0]

    # Ensure locked players are available
    missing_locks = set(locked_player_ids) - set(df["player_id"].tolist())
    if missing_locks:
        raise ValueError(f"Locked players not found in slate: {', '.join(missing_locks)}")

    return df


def _build_correlated_lineup(
    pool: pd.DataFrame,
    template: LineupTemplate,
    locked_ids: List[str],
    game_envs: Dict[str, Dict[str, Any]],
    team_stacks: Dict[str, List[Dict[str, Any]]],
    correlation_strength: float = 0.7,
    max_salary_leaving: int = 1000,
    rng: Optional[random.Random] = None,
) -> Optional[Tuple[List[str], float]]:
    """
    Generate a single correlated lineup with stacking logic.
    
    Returns: (lineup_player_ids, correlation_score) or None
    
    correlation_strength: 0.0 = pure random, 1.0 = maximum correlation
    """
    if rng is None:
        rng = random
    
    roster_size = template.roster_size
    cap = template.salary_cap
    
    # Start with locked players
    current_ids = list(dict.fromkeys(locked_ids))
    
    # Check locks validity
    locks_df = pool[pool["player_id"].isin(current_ids)]
    lock_salary = int(locks_df["salary"].sum())
    if len(current_ids) > roster_size or lock_salary > cap:
        return None
    
    remaining_spots = roster_size - len(current_ids)
    remaining_cap = cap - lock_salary
    
    # Available pool
    available = pool[~pool["player_id"].isin(current_ids)].copy()
    if available.empty and remaining_spots > 0:
        return None
    
    # Decide on stack strategy based on correlation_strength
    use_stack = rng.random() < correlation_strength
    
    if use_stack and remaining_spots >= 2:
        # Try to build a correlated stack
        
        # Priority 1: Pick a high-upside game
        high_upside_games = [
            gid for gid, ginfo in game_envs.items()
            if ginfo.get("is_high_upside", False)
        ]
        
        if high_upside_games:
            # Weighted selection of game
            game_weights = [game_envs[g]["quality_score"] for g in high_upside_games]
            total_weight = sum(game_weights)
            if total_weight > 0:
                game_probs = [w / total_weight for w in game_weights]
                target_game = rng.choices(high_upside_games, weights=game_probs)[0]
            else:
                target_game = rng.choice(high_upside_games)
            
            # Priority 2: Select a team stack from that game
            game_teams = game_envs[target_game]["teams"]
            primary_team = rng.choice(game_teams)
            
            # Try to find a team stack
            if primary_team in team_stacks and team_stacks[primary_team]:
                # Pick a stack (weighted by stack_score)
                stacks = team_stacks[primary_team]
                stack_weights = [s["stack_score"] for s in stacks]
                total_sw = sum(stack_weights)
                
                if total_sw > 0:
                    stack_probs = [w / total_sw for w in stack_weights]
                    chosen_stack = rng.choices(stacks, weights=stack_probs)[0]
                    
                    # Add primary and secondary players
                    stack_players = [chosen_stack["primary"], chosen_stack["secondary"]]
                    
                    # Check affordability
                    stack_df = available[available["player_id"].isin(stack_players)]
                    stack_salary = int(stack_df["salary"].sum())
                    
                    if len(stack_df) == 2 and stack_salary <= remaining_cap:
                        current_ids.extend(stack_players)
                        remaining_cap -= stack_salary
                        remaining_spots -= 2
                        
                        # Priority 3: Add a bring-back if room remains
                        if remaining_spots >= 1:
                            bringback_candidates = identify_bringback_candidates(
                                available, primary_team, target_game
                            )
                            
                            bringback_candidates = [
                                pid for pid in bringback_candidates
                                if pid not in current_ids
                            ]
                            
                            if bringback_candidates:
                                # Try each candidate
                                rng.shuffle(bringback_candidates)
                                for bb_id in bringback_candidates:
                                    bb_player = available[available["player_id"] == bb_id]
                                    if not bb_player.empty:
                                        bb_salary = int(bb_player["salary"].iloc[0])
                                        if bb_salary <= remaining_cap:
                                            current_ids.append(bb_id)
                                            remaining_cap -= bb_salary
                                            remaining_spots -= 1
                                            break
    
    # Fill remaining spots with value-based random selection
    available = pool[~pool["player_id"].isin(current_ids)].copy()
    
    if remaining_spots > 0 and not available.empty:
        # Value-weighted selection
        if "value" in available.columns:
            base_score = available["value"].values
        else:
            base_score = available["proj"].values / np.maximum(available["salary"].values / 1000, 1)
        
        scores = []
        for v in base_score:
            noise = rng.normalvariate(0, 0.2)
            scores.append(v + noise)
        
        available = available.assign(_score=scores).sort_values("_score", ascending=False)
        
        for _, row in available.iterrows():
            if remaining_spots == 0:
                break
            if row["salary"] <= remaining_cap:
                current_ids.append(row["player_id"])
                remaining_spots -= 1
                remaining_cap -= int(row["salary"])
    
    if remaining_spots != 0:
        return None
    
    # Check salary usage
    used_salary = cap - remaining_cap
    if cap - used_salary > max_salary_leaving:
        return None
    
    # Calculate correlation score for this lineup
    lineup_df = pool[pool["player_id"].isin(current_ids)]
    corr_score = calculate_lineup_correlation_score(lineup_df, game_envs)
    
    return (current_ids, corr_score)


def generate_top_n_lineups(
    slate_df: pd.DataFrame,
    template: LineupTemplate,
    n_lineups: int = 10,
    correlation_strength: float = 0.7,  # NEW: 0.0-1.0, how aggressive to stack
    locked_player_ids: Optional[List[str]] = None,
    excluded_player_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a set of high-projection, salary-legal lineups with correlation.
    
    NEW in enhanced version:
    - Builds game and team stacks for tournament upside
    - Uses bring-back logic for game correlation
    - Balances projection with correlation score
    - correlation_strength parameter controls aggression (0.7+ for GPPs)

    Returns a list of dicts:
        {
            'player_ids': [str, ...],
            'proj_score': float,
            'salary_used': int,
            'correlation_score': float,  # NEW
            'num_games': int,  # NEW
            'num_teams': int,  # NEW
        }
    """
    locked_player_ids = locked_player_ids or []
    excluded_player_ids = excluded_player_ids or []

    pool = _prepare_player_pool(slate_df, locked_player_ids, excluded_player_ids)
    
    # Build correlation metadata
    game_envs = build_game_environments(pool)
    team_stacks = build_team_stacks(pool)

    rng = random.Random(42)

    best: Dict[frozenset, Dict[str, Any]] = {}

    # More attempts for correlated building (it's more constrained)
    attempts = max(3000, 500 * n_lineups)
    
    for _ in range(attempts):
        result = _build_correlated_lineup(
            pool, template, locked_player_ids, game_envs, team_stacks,
            correlation_strength=correlation_strength, rng=rng
        )
        
        if result is None:
            continue
        
        lineup_ids, corr_score = result
        key = frozenset(lineup_ids)
        
        if key in best:
            continue

        lineup_df = pool[pool["player_id"].isin(lineup_ids)]
        proj_score = float(lineup_df["proj"].sum())
        salary_used = int(lineup_df["salary"].sum())
        
        # Count unique games and teams (diversity metrics)
        num_games = lineup_df["GameID"].nunique()
        num_teams = lineup_df["Team"].nunique()

        best[key] = {
            "player_ids": lineup_ids,
            "proj_score": proj_score,
            "salary_used": salary_used,
            "correlation_score": corr_score,
            "num_games": num_games,
            "num_teams": num_teams,
        }

    # Sort by composite score: projection + correlation
    # Weight correlation more heavily for GPP builds
    all_lineups = list(best.values())
    
    for lu in all_lineups:
        # Normalize correlation score (0-100 range typically)
        norm_corr = lu["correlation_score"] / 100.0 if lu["correlation_score"] > 0 else 0
        
        # Composite score blends projection and correlation
        # Higher correlation_strength = more weight on correlation
        lu["composite_score"] = (
            lu["proj_score"] * (1.0 - correlation_strength * 0.3) +
            norm_corr * 50 * correlation_strength
        )
    
    all_lineups = sorted(all_lineups, key=lambda x: x["composite_score"], reverse=True)
    return all_lineups[:n_lineups]
