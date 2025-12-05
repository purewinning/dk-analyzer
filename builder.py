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
# Note: UTIL is implicitly covered by TOTAL_PLAYERS.
ROSTER_REQUIREMENTS = {
    'PG': 1,
    'SG': 1,
    'SF': 1,
    'PF': 1,
    'C': 1,
    'G': 1,  # Guard (PG or SG)
    'F': 1,  # Forward (SF or PF)
}

MIN_GAMES_REQUIRED = 2 # DK rule: Lineups must include players from at least 2 different games.


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
    """Defines the target ownership structure for a contest type."""
    contest_label: str
    roster_size: int
    salary_cap: int
    target_mega: float
    target_chalk: float
    target_mid: float
    target_punt: float
    # Positional/Slate Rules (can be customized for NFL/Showdown later)
    roster_slots: Dict[str, int] = field(default_factory=lambda: ROSTER_REQUIREMENTS)
    min_games: int = MIN_GAMES_REQUIRED

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
    roster_slots: Dict[str, int] = ROSTER_REQUIREMENTS, # New: Pass in specific positional requirements
    min_games: int = MIN_GAMES_REQUIRED, # New: Pass in min games
) -> StructureTemplate:
    """
    Heuristic mapping: contest traits → ideal ownership structure.
    (Existing logic preserved)
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

    # ... (Rest of the contest type logic remains the same for ownership targets) ...
    # This block is the same as in your original builder.py

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
