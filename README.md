# NBA DFS Lineup Builder - Enhanced with Correlation & Stacking

## 🎯 What This Does

This enhanced version of your NBA DFS lineup builder adds **tournament-grade correlation logic** to create lineups with stronger upside edges. Instead of just picking the highest projected players, it now builds **strategically correlated lineups** using game stacks, team stacks, and bring-back strategies.

## 🚀 Key Enhancements

### 1. **Game Stacking**
- Identifies high-scoring game environments
- Groups players from the same game to amplify upside
- If a game goes over, multiple lineup spots benefit simultaneously

### 2. **Team Stacking**  
- Pairs stars with value teammates from the same team
- Creates unique exposures (e.g., 40% star + 10% role player = 4% combined)
- Leverages natural team correlation (when star succeeds, team succeeds)

### 3. **Bring-Back Logic**
- Adds opposing players to game stacks
- Hedges while maintaining correlation
- Classic GPP strategy: "Stack one team, bring back the other"

### 4. **Correlation Scoring**
- Every lineup gets a correlation strength score (0-100)
- Balances projection with correlation based on contest type
- Composite scoring: `projection * (1-α) + correlation * α`

### 5. **Contest-Adaptive Defaults**
- Cash Games: Low correlation (0.2) - prioritize floor
- Single Entry: Moderate (0.5) - balanced
- 20-Max: High (0.75) - need differentiation  
- Milly Maker: Maximum (0.85) - all upside

## 📁 Files Included

### Core Files
- **`builder_enhanced.py`** - Enhanced lineup generation with correlation logic
- **`app_enhanced.py`** - Updated Streamlit UI with correlation controls

### Documentation  
- **`CORRELATION_GUIDE.md`** - Deep dive into correlation concepts and implementation
- **`MIGRATION_GUIDE.md`** - How to upgrade from your original version
- **`QUICK_REFERENCE.md`** - Cheat sheet for correlation strategies
- **`README.md`** - This file

## 🔧 Installation

### Option 1: Replace Existing Files
```bash
# Backup your originals first!
cp builder.py builder_backup.py
cp app.py app_backup.py

# Replace with enhanced versions
cp builder_enhanced.py builder.py
cp app_enhanced.py app.py
```

### Option 2: Run Side-by-Side
```bash
# Keep both versions
streamlit run app.py              # Original
streamlit run app_enhanced.py     # Enhanced
```

### Requirements
Same as your original:
```
streamlit
pandas
numpy
```

## 🎮 Usage

### Basic Usage (Unchanged)
1. Paste your DraftKings player pool CSV/TSV
2. Select contest type
3. Lock/Exclude players as desired
4. Generate lineups

### New Controls

#### Correlation Strength Slider
- **Location:** Sidebar, below "Number of Lineups"
- **Range:** 0.0 (none) to 1.0 (maximum)
- **Auto-sets based on contest type**

**Recommendations:**
- Cash: 0.2 (light correlation for safety)
- GPPs under 1000 entries: 0.5-0.6
- GPPs 1000-10000: 0.7-0.8  
- Large GPPs (10k+): 0.8-0.9

### New Display Features

#### Lineup Summary
- **Corr column** - Correlation score (higher = more stacked)
- **Games column** - Number of unique games represented
- **Teams column** - Number of unique teams represented

#### Lineup Detail  
- **Stack information** - Shows team stacks (e.g., "BOS (3x)")
- **Game stacks** - Highlights multi-player game correlations

## 💡 Quick Start Guide

### For Your First Build

1. **Load your data** (same CSV format as before)

2. **Select contest type** (this auto-sets correlation)

3. **Generate lineups** with default settings

4. **Review the "Corr" column** in lineup summary:
   - 40-60: Moderate correlation
   - 60-80: Strong tournament structure
   - 80+: Maximum upside play

5. **Check lineup detail** to see the actual stacks

### Example: 20-Max GPP

```
Settings:
- Contest: 20-Max
- Correlation: 0.75 (default)
- Lineups: 20

Expected Output:
- Mix of correlation scores (50-85)
- 2-4 players from primary game
- 6-7 unique teams per lineup
- 3-5 unique games per lineup
```

### Example: Cash Game

```
Settings:
- Contest: Cash Game
- Correlation: 0.2 (default)
- Lineups: 1

Expected Output:
- Low correlation score (20-40)
- Mostly spread across games
- Focus on projection and floor
```

## 🎯 Strategy Tips

### When to Use HIGH Correlation (0.7+)

✅ Large-field GPPs (5000+ entries)
✅ Need differentiation from field
✅ Top-heavy prize structures
✅ When you identify low-owned game environments

### When to Use MODERATE Correlation (0.4-0.6)

✅ Single entry contests  
✅ Small GPPs (under 1000)
✅ Want balance of floor and ceiling
✅ First time using correlation

### When to Use LOW Correlation (0.2-0.3)

✅ Cash games always
✅ When locks break natural stacks  
✅ Conservative GPP approach
✅ Slate doesn't have clear high-upside games

## 📊 Understanding the Output

### Correlation Score Ranges

| Score | Interpretation | Use Case |
|-------|----------------|----------|
| 0-30 | Weak/No correlation | Cash games |
| 30-50 | Light correlation | Conservative GPPs |
| 50-70 | Good GPP structure | Most GPPs |
| 70-85 | Strong tournament play | Large-field GPPs |
| 85+ | Maximum upside | Contrarian/Milly |

### Example Lineup Analysis

```
Lineup #1
Proj: 285.5
Corr: 72.8
Games: 4
Teams: 6

Detail:
- BOS (3 players) - Tatum, Brown, Horford
- LAL (2 players) - Davis, Reaves  
- DEN (1 player) - Jokic
- OKC (1 player) - SGA
- MIL (1 player) - Giannis

Analysis:
✅ Strong BOS@LAL game stack (5 players)
✅ High correlation score
✅ Diversified with studs from other games
✅ Good tournament structure
```

## 🔬 Technical Deep Dive

### How Correlation Works

**Step 1: Game Environment Analysis**
```python
# Identifies high-scoring games
# Calculates quality score based on:
# - Projected total
# - Spread (closer = better)
# - Player ceiling averages
```

**Step 2: Team Stack Library**
```python
# For each team:
# - Identify stars (top 30% salary)
# - Identify value plays (good value metric)
# - Score all star+value combinations
# - Rank by combined ceiling and ownership differentiation
```

**Step 3: Lineup Construction**
```python
# Weighted random process:
# 1. Select high-quality game (weighted by quality_score)
# 2. Select team stack from that game (weighted by stack_score)  
# 3. Add bring-back from opposing team
# 4. Fill remaining spots with value
# 5. Validate salary and roster constraints
```

**Step 4: Scoring**
```python
# Each lineup gets two scores:
# - Projection score (fantasy points)
# - Correlation score (stack strength)
# 
# Final ranking:
# composite = proj * (1 - α*0.3) + corr * 50 * α
# where α = correlation_strength setting
```

### Key Functions

- `build_game_environments()` - Analyzes slate for high-upside games
- `build_team_stacks()` - Creates star+value stack library  
- `identify_bringback_candidates()` - Finds optimal opposing players
- `_build_correlated_lineup()` - Generates single correlated lineup
- `calculate_lineup_correlation_score()` - Scores lineup correlation strength

## 🐛 Troubleshooting

### "No valid lineups generated"

**Likely causes:**
1. Correlation too high - try lowering to 0.5-0.6
2. Too many locked players - unlock 1-2
3. Locks spread across too many games
4. Small player pool (need 20+ for good stacking)

### "All lineups look similar"

**This is normal when:**
- Correlation is very high (0.85+)
- Slate has one dominant stack
- Builder found the optimal structure

**Solutions:**
- Lower correlation slightly
- Generate more lineups (30-40)
- Accept some clustering (optimal plays converge)

### "Correlation scores are low"

**Possible reasons:**
- Slate structure (no clear high-upside games)
- Your locks are dispersed
- Correlation strength set too low
- This is okay! Not every slate offers strong correlation

## 📈 Measuring Success

After contests run, compare:

1. **Your best correlated lineup vs your best non-correlated**
2. **Win rate of high-corr (70+) vs low-corr (30-) lineups**
3. **Which correlation level worked for your contest type**

Iterate and refine your correlation preferences!

## 🔄 Rollback Option

Not working for you? Easy rollback:

```bash
# Restore originals
cp builder_backup.py builder.py
cp app_backup.py app.py
```

Your original files are unchanged and ready to use.

## 📚 Further Reading

- **`CORRELATION_GUIDE.md`** - Full explanation of correlation theory
- **`MIGRATION_GUIDE.md`** - Detailed upgrade instructions  
- **`QUICK_REFERENCE.md`** - Strategy cheat sheet

## 🤝 Credits

**Original Builder:** Your DFS lineup optimizer with projection/edge logic

**Enhanced By:** Adding tournament correlation and stacking strategies

**Philosophy:** "In GPPs, unique structure beats raw projection"

## 📝 Version History

### v2.0 - Enhanced (This Version)
- ✅ Game stack analysis
- ✅ Team stack identification  
- ✅ Bring-back logic
- ✅ Correlation scoring
- ✅ Contest-adaptive correlation
- ✅ Enhanced UI with correlation metrics

### v1.0 - Original
- ✅ Projection-based optimization
- ✅ GPP score calculation
- ✅ Ownership consideration
- ✅ Lock/Exclude functionality
- ✅ Multi-lineup generation

## 🎯 Final Thoughts

**This enhancement doesn't replace your original builder** - it augments it with powerful correlation logic for tournaments. 

For cash games, it behaves almost identically (low correlation).

For GPPs, it unlocks structured upside that helps you:
- ✅ Differentiate from the field
- ✅ Build tournament-winning ceiling
- ✅ Create unique exposures
- ✅ Leverage game/team correlation

**Try it on your next slate and watch the correlation scores!** 🚀

---

*For questions or issues, refer to the documentation files or dial back the correlation strength to see more traditional builds.*
