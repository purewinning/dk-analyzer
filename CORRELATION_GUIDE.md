# NBA DFS Lineup Builder - Correlation & Stacking Enhancements

## Overview

This enhanced version adds **tournament-grade correlation logic** to build lineups with stronger upside edges. The key philosophy: in GPPs, you need **differentiated, correlated upside** to finish at the top, not just the highest raw projection.

## What's New

### 1. **Game Environment Analysis**
```python
build_game_environments(df)
```

**What it does:**
- Identifies high-scoring game environments
- Calculates game totals and spreads
- Scores games based on upside potential
- Flags "blowout risk" games (correlation breaks down in blowouts)

**Why it matters:**
- Players from the same high-scoring game are naturally correlated
- If a game goes over, multiple players benefit
- This is a foundational edge in tournament play

**Example Output:**
```python
{
    "BOS@LAL": {
        "teams": ["BOS", "LAL"],
        "total_proj": 420.5,
        "spread": 12.3,
        "quality_score": 398.2,
        "avg_ceiling": 55.6,
        "is_high_upside": True
    }
}
```

---

### 2. **Team Stacking**
```python
build_team_stacks(df)
```

**What it does:**
- Identifies viable primary + secondary player combos from same team
- Scores stacks based on:
  - Combined ceiling potential
  - Combined ownership (lower = better differentiation)
  - Value efficiency

**Why it matters:**
- When a star goes off, their teammates often benefit (more possessions, blowout minutes, etc.)
- Pairing chalk stars with low-owned role players creates unique exposures
- Classic GPP strategy: "Stars + Value from same team"

**Example Stack:**
```python
{
    "primary": "Jayson Tatum",      # 35% owned star
    "secondary": "Derrick White",   # 8% owned value
    "combined_own": 43%,
    "combined_ceiling": 108.5,
    "stack_score": 87.3
}
```

---

### 3. **Bring-Back Logic**
```python
identify_bringback_candidates(df, primary_team, game_id)
```

**What it does:**
- When you stack a team, identifies optimal opposing players to pair
- Scores candidates on ceiling, ownership, and value

**Why it matters:**
- **Game stack theory**: If BOS vs LAL goes over, both teams score
- Instead of just "BOS stack," you want "BOS stack + LAL bringback"
- This creates **game-correlated lineups** with differentiated structure

**Example:**
- Stack: Tatum (BOS) + Brown (BOS)
- Bring-back: Anthony Davis (LAL)
- If the game explodes, you have exposure to both sides

---

### 4. **Correlation Scoring**
```python
calculate_lineup_correlation_score(lineup_df, game_envs)
```

**What it does:**
- Scores entire lineups based on correlation strength
- Bonuses for:
  - Multiple players from same game
  - Multiple players from same team
  - High-quality game environments
- Penalties for:
  - Too many different games (diluted correlation)

**Why it matters:**
- Balances projection with correlation
- Ensures lineups aren't just "8 random high-value guys"
- Creates **structured upside** rather than hope

---

### 5. **Composite Scoring**

The final lineup scoring combines:

```python
composite_score = (
    proj_score * (1.0 - correlation_strength * 0.3) +
    correlation_score * 50 * correlation_strength
)
```

**Variables:**
- `proj_score`: Raw fantasy point projection
- `correlation_score`: Stack/game correlation strength (0-100)
- `correlation_strength`: User-controlled slider (0.0-1.0)

**Contest-Specific Defaults:**
- Cash Games: 0.2 (light correlation for safety)
- Single Entry: 0.5 (balanced)
- 3-Max: 0.6 (more aggressive)
- 20-Max: 0.75 (high correlation)
- Milly Maker: 0.85 (maximum stacking)

---

## How the Builder Works Now

### Old Approach (Original):
1. Generate random lineups
2. Score by projection + GPP score - ownership
3. Keep top N by projection

**Problem:** No correlation → low ceiling, hard to differentiate

---

### New Approach (Enhanced):

1. **Analyze slate structure**
   - Identify high-upside games
   - Build team stack library
   - Score game environments

2. **Build with correlation**
   - Select high-upside game (weighted by quality)
   - Select team stack from that game (weighted by stack score)
   - Add bring-back from opposing team
   - Fill remaining spots with value

3. **Score & rank**
   - Calculate projection (still matters!)
   - Calculate correlation score
   - Blend based on contest type
   - Return top N by composite score

---

## Key Parameters

### `correlation_strength` (0.0 - 1.0)

**What it controls:**
- How aggressively the builder pursues stacks
- Weight given to correlation vs raw projection

**Settings:**
- **0.0-0.3**: Balanced/Cash - Prioritizes projection, light stacking
- **0.4-0.6**: Moderate - Good for SE/3-Max
- **0.7-0.9**: Aggressive - Heavy stacking for large-field GPPs
- **0.9-1.0**: Maximum - Extreme correlation (use sparingly)

**Example:**
- At 0.5: Lineups might have 1-2 mini-stacks
- At 0.8: Lineups will have 2-3 player game stacks + bring-backs

---

## Output Enhancements

Each lineup now includes:

```python
{
    'player_ids': [...],
    'proj_score': 285.5,
    'salary_used': 49800,
    'correlation_score': 73.2,      # NEW
    'num_games': 4,                  # NEW
    'num_teams': 6,                  # NEW
    'composite_score': 312.7         # NEW
}
```

**New UI Display:**
- Shows correlation score in lineup summary
- Shows num games/teams (diversity metrics)
- Displays stack information (e.g., "BOS (3x), LAL (2x)")
- Game stack indicators

---

## Strategic Insights

### When to Use High Correlation (0.7+)

✅ **Large-field GPPs (5000+ entries)**
- Need unique lineup structures to differentiate
- Top-heavy prize structures reward ceiling over consistency

✅ **Late swap slates**
- Can pivot to confirmed high-scoring games
- Real-time information creates edges

✅ **Lower-owned games**
- If BOS@LAL is popular but DEN@PHX isn't, stack DEN@PHX
- Correlation + low ownership = massive leverage

### When to Use Moderate Correlation (0.4-0.6)

✅ **Single entry contests**
- One lineup, need balance of floor and ceiling

✅ **Small-field GPPs (under 1000)**
- Less need for extreme differentiation
- Projection matters more

✅ **3-Max/20-Max**
- Want some diversity across your lineups
- Not all need maximum correlation

### When to Use Low Correlation (0.2-0.3)

✅ **Cash games**
- Floor > ceiling
- Avoid correlated downside
- Want consistent, safe plays

✅ **When locks break stacks**
- If you're locking in players from 5 different games, forced correlation won't help

---

## Edge Building Philosophy

### The Math Behind Stacks

**Scenario:** BOS@LAL game
- Projected total: 230 points
- Actual if it goes over (+10%): 253 points

**Non-correlated lineup:**
- Has Tatum from this game
- Other 7 players from different games
- If BOS@LAL goes over, 1/8 of lineup benefits

**Correlated lineup:**
- Has Tatum + Brown (BOS)
- Has Davis (LAL) as bring-back
- If BOS@LAL goes over, 3/8 of lineup benefits (2.4x multiplier)

**Result:** The correlated lineup has **amplified upside** if the game hits.

---

### Ownership Leverage Through Stacks

**Common mistake:** "Jokic is 40% owned so I'll fade him"

**Better approach:** "Jokic is 40% owned, but stacking him with MPJ (12% owned) creates a 40% × 12% = 4.8% combined exposure that the field doesn't have"

**This builder does this automatically:**
- Finds high-leverage stack combinations
- Pairs popular players with unpopular teammates
- Creates unique structures even with chalky cores

---

## Example Lineups

### High Correlation (0.8) - Milly Maker Style
```
Game Stack: BOS@LAL (4 players)
- Tatum (BOS)
- Brown (BOS)
- Davis (LAL)
- Reaves (LAL)

Game Stack: DEN@PHX (2 players)
- Jokic (DEN)
- Murray (DEN)

Fill: 
- Curry (GSW)
- Doncic (DAL)

Correlation Score: 82.3
Games: 4
Teams: 6
```

### Moderate Correlation (0.5) - Single Entry Style
```
Mini Stack: MIL (2 players)
- Giannis
- Lillard

Bring-back: IND
- Haliburton

Fill (value):
- SGA (OKC)
- Maxey (PHI)
- Fox (SAC)
- Sengun (HOU)
- Claxton (BKN)

Correlation Score: 45.8
Games: 6
Teams: 8
```

---

## Technical Implementation

### Core Functions

1. **`build_game_environments()`**
   - Input: Player pool DataFrame
   - Output: Dict of game metadata
   - Called: Once per slate load

2. **`build_team_stacks()`**
   - Input: Player pool DataFrame
   - Output: Dict of viable stacks by team
   - Called: Once per slate load

3. **`_build_correlated_lineup()`**
   - Input: Pool, locks, correlation strength
   - Output: Single lineup with correlation score
   - Called: 3000+ times per build (stochastic search)

4. **`calculate_lineup_correlation_score()`**
   - Input: Completed lineup
   - Output: Correlation strength score
   - Called: For each generated lineup

### Performance

- **Old builder:** ~2000 attempts to generate 20 lineups
- **New builder:** ~3000+ attempts (more constrained)
- **Runtime:** Still sub-second for most builds
- **Quality:** Significantly higher ceiling potential

---

## Usage Tips

### For GPPs:
1. Set correlation to 0.7-0.85
2. Look for low-owned game stacks
3. Use the "Edge Category" column to find leverage
4. Lock 1-2 contrarian plays, let builder find the stack

### For Cash:
1. Set correlation to 0.2-0.3
2. Lock your chalky floor plays
3. Let builder fill with value
4. Ignore correlation score, focus on projection

### For Differentiation:
1. Identify the popular game stack (e.g., everyone on BOS@LAL)
2. Use the builder to find the counter-stack
3. If BOS@LAL is 60% game stack rate, go DEN@PHX
4. You'll be correlated differently than the field

---

## Future Enhancements (Possible)

- **Position-specific correlation** (PG + C from same team often correlate)
- **Pace-based game selection** (favor fast-paced games)
- **Leverage-weighted stacking** (prioritize stacks with leverage, not just ceiling)
- **Anti-correlation** (fade correlated chalk when appropriate)
- **Game theory analysis** (simulate field ownership of stacks)

---

## Questions?

**Q: Should I always use high correlation?**
A: No. In small fields or cash, projection > correlation. Use high correlation only when you need ceiling/differentiation.

**Q: Can I still lock players?**
A: Yes! The builder respects locks and builds stacks around them when possible.

**Q: What if my locks break the correlation?**
A: The builder will still try to stack non-locked players. But if you lock 5 players from 5 different games, correlation won't help much.

**Q: How do I know if a lineup is "good"?**
A: Look at correlation score + projection together. For GPPs, 70+ correlation with 280+ projection is strong.

**Q: Does this work for other sports?**
A: The concepts (game stacks, bring-backs) work for NFL, NHL. MLB is different (pitcher stacking). Would need sport-specific tuning.

---

## Summary

**Old builder:** Maximized projection with some GPP scoring adjustments

**New builder:** Maximizes **correlated upside** by intelligently stacking players from high-scoring games and teams, creating tournament-winning lineup structures

**Key insight:** In GPPs, it's not just about having the highest projected score — it's about having a **unique structure** that capitalizes on **correlated outcomes** the field doesn't have.

This is how you build edges that ship tournaments. 🏆
