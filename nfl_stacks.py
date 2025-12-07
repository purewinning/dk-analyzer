# NFL-specific stacking and correlation logic

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import random


def build_nfl_stacks(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build NFL-specific stacks:
    1. QB + Pass Catcher stacks
    2. RB + DST same-team stacks
    3. Game stacks with bring-backs
    """
    qb_stacks = {}
    rb_dst_stacks = {}
    
    for team in df["Team"].unique():
        team_players = df[df["Team"] == team].copy()
        
        # QB Stacks
        qbs = team_players[team_players["positions"].str.contains("QB", case=False, na=False)]
        pass_catchers = team_players[
            team_players["positions"].str.contains("WR|TE", case=False, na=False)
        ]
        
        for _, qb in qbs.iterrows():
            stacks = []
            for _, pc in pass_catchers.iterrows():
                combined_own = qb["own_proj"] + pc["own_proj"]
                combined_ceiling = qb["ceiling"] + pc["ceiling"]
                combined_proj = qb["proj"] + pc["proj"]
                
                ownership_diff = 100 - combined_own
                
                stack_score = (
                    combined_ceiling * 0.4 +
                    combined_proj * 0.4 +
                    ownership_diff * 0.2
                )
                
                stacks.append({
                    "qb": qb["player_id"],
                    "pass_catcher": pc["player_id"],
                    "combined_own": combined_own,
                    "combined_ceiling": combined_ceiling,
                    "stack_score": stack_score,
                    "type": "qb_pass_catcher"
                })
            
            qb_stacks[team] = sorted(stacks, key=lambda x: x["stack_score"], reverse=True)[:15]
        
        # RB + DST Stacks
        rbs = team_players[team_players["positions"].str.contains("RB", case=False, na=False)]
        dsts = team_players[team_players["positions"].str.contains("DST|DEF", case=False, na=False)]
        
        if not rbs.empty and not dsts.empty:
            rb_dst_combos = []
            for _, rb in rbs.iterrows():
                for _, dst in dsts.iterrows():
                    combined_own = rb["own_proj"] + dst["own_proj"]
                    combined_proj = rb["proj"] + dst["proj"]
                    
                    # RB+DST correlation is strong in positive game scripts
                    stack_score = (
                        combined_proj * 0.5 +
                        (100 - combined_own) * 0.3 +
                        rb["value"] * 5
                    )
                    
                    rb_dst_combos.append({
                        "rb": rb["player_id"],
                        "dst": dst["player_id"],
                        "combined_own": combined_own,
                        "stack_score": stack_score,
                        "type": "rb_dst"
                    })
            
            rb_dst_stacks[team] = sorted(rb_dst_combos, key=lambda x: x["stack_score"], reverse=True)[:5]
    
    return {"qb_stacks": qb_stacks, "rb_dst_stacks": rb_dst_stacks}


def identify_nfl_bringback(
    df: pd.DataFrame,
    primary_team: str,
    game_id: str,
    primary_positions: List[str],
    max_candidates: int = 5
) -> List[str]:
    """
    Find NFL bring-back players from opposing team.
    
    If stacking QB+WR, bring back opposing pass catcher or QB.
    If stacking RB+DST, bring back opposing QB or pass catcher.
    """
    game_teams = df[df["GameID"] == game_id]["Team"].unique()
    opposing_team = [t for t in game_teams if t != primary_team]
    
    if not opposing_team:
        return []
    
    opposing_team = opposing_team[0]
    candidates = df[df["Team"] == opposing_team].copy()
    
    if candidates.empty:
        return []
    
    # Determine bring-back positions based on primary stack
    if "QB" in primary_positions or "WR" in primary_positions or "TE" in primary_positions:
        # Offensive stack - bring back opposing offense
        candidates = candidates[
            candidates["positions"].str.contains("QB|WR|TE", case=False, na=False)
        ]
    else:
        # RB or DST stack - bring back any high-value player
        candidates = candidates[
            candidates["positions"].str.contains("QB|WR|RB|TE", case=False, na=False)
        ]
    
    if candidates.empty:
        return []
    
    # Score bring-back candidates
    candidates["bringback_score"] = (
        candidates["ceiling"] * 0.4 +
        (100 - candidates["own_proj"]) * 0.3 +
        candidates["value"] * 10 * 0.3
    )
    
    top_candidates = candidates.nlargest(max_candidates, "bringback_score")
    return top_candidates["player_id"].tolist()


def validate_nfl_lineup(lineup_df: pd.DataFrame) -> bool:
    """
    Check for NFL anti-correlations that should be avoided.
    Returns True if lineup is valid, False if has bad correlations.
    """
    # Check for QB + opposing DST (bad correlation)
    qbs = lineup_df[lineup_df["positions"].str.contains("QB", case=False, na=False)]
    dsts = lineup_df[lineup_df["positions"].str.contains("DST|DEF", case=False, na=False)]
    
    for _, qb in qbs.iterrows():
        qb_team = qb["Team"]
        qb_opp = qb["Opponent"]
        
        for _, dst in dsts.iterrows():
            dst_team = dst["Team"]
            # QB facing DST's team is bad
            if qb_opp == dst_team:
                return False
    
    # Check for WR/TE + opposing DST (bad correlation)
    pass_catchers = lineup_df[lineup_df["positions"].str.contains("WR|TE", case=False, na=False)]
    
    for _, pc in pass_catchers.iterrows():
        pc_opp = pc["Opponent"]
        for _, dst in dsts.iterrows():
            dst_team = dst["Team"]
            if pc_opp == dst_team:
                return False
    
    # Check for same-team RB1 + RB2 (TD competition - generally avoided)
    rbs = lineup_df[lineup_df["positions"].str.contains("RB", case=False, na=False)]
    if len(rbs) >= 2:
        teams = rbs["Team"].tolist()
        if len(teams) != len(set(teams)):
            # Has duplicate team in RB positions
            # This is sometimes OK (game script hedge) but flag it
            # For now we'll allow it
            pass
    
    return True


def calculate_nfl_correlation_score(
    lineup_df: pd.DataFrame,
    game_envs: Dict[str, Dict[str, Any]]
) -> float:
    """
    Score NFL lineup based on correlation strength.
    Higher score = better NFL-specific correlation.
    """
    if lineup_df.empty:
        return 0.0
    
    score = 0.0
    
    # QB + Pass Catcher same team (primary stack)
    qbs = lineup_df[lineup_df["positions"].str.contains("QB", case=False, na=False)]
    pass_catchers = lineup_df[lineup_df["positions"].str.contains("WR|TE", case=False, na=False)]
    
    for _, qb in qbs.iterrows():
        qb_team = qb["Team"]
        same_team_pc = pass_catchers[pass_catchers["Team"] == qb_team]
        
        if len(same_team_pc) >= 1:
            score += 30  # QB + 1 pass catcher
        if len(same_team_pc) >= 2:
            score += 20  # QB + 2 pass catchers (mini-stack)
        if len(same_team_pc) >= 3:
            score += 15  # QB + 3 pass catchers (full stack)
    
    # RB + DST same team (game script correlation)
    rbs = lineup_df[lineup_df["positions"].str.contains("RB", case=False, na=False)]
    dsts = lineup_df[lineup_df["positions"].str.contains("DST|DEF", case=False, na=False)]
    
    for _, rb in rbs.iterrows():
        rb_team = rb["Team"]
        same_team_dst = dsts[dsts["Team"] == rb_team]
        if len(same_team_dst) > 0:
            score += 25  # RB + DST correlation
    
    # Game stack bonus (multiple players from same game)
    game_counts = lineup_df.groupby("GameID").size()
    for game_id, count in game_counts.items():
        if count >= 3:
            game_info = game_envs.get(game_id, {})
            quality = game_info.get("quality_score", 0)
            score += (count - 2) * quality * 0.08
    
    # Bring-back bonus (players from both teams in same game)
    for game_id in lineup_df["GameID"].unique():
        game_players = lineup_df[lineup_df["GameID"] == game_id]
        teams_in_game = game_players["Team"].nunique()
        if teams_in_game == 2:
            score += 15  # Has bring-back
    
    return score
