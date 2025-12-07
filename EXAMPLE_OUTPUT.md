# Example Enhanced Lineup Output

## Sample Lineup Comparison: Original vs Enhanced

### Original Builder Output (v1.0)

```
Lineup #1 - Projection: 287.3
┌──────┬──────────────────┬──────┬────────┬──────────┬─────────┬───────┐
│ SLOT │ Player           │ Pos  │ Team   │ Opponent │ Salary  │ Proj  │
├──────┼──────────────────┼──────┼────────┼──────────┼─────────┼───────┤
│ PG   │ Tyrese Haliburton│ PG   │ IND    │ DET      │ $9,200  │ 48.5  │
│ SG   │ Dejounte Murray  │ PG/SG│ NOP    │ CHI      │ $7,800  │ 38.2  │
│ SF   │ Kawhi Leonard    │ SF/PF│ LAC    │ POR      │ $8,900  │ 42.1  │
│ PF   │ Pascal Siakam    │ PF   │ IND    │ DET      │ $8,400  │ 39.8  │
│ C    │ Nikola Jokic     │ C    │ DEN    │ PHX      │ $11,500 │ 58.9  │
│ G    │ Jalen Brunson    │ PG/SG│ NYK    │ BKN      │ $8,700  │ 41.3  │
│ F    │ Jimmy Butler     │ SF/PF│ MIA    │ ORL      │ $7,600  │ 36.8  │
│ UTIL │ Isaiah Stewart   │ PF/C │ DET    │ IND      │ $4,900  │ 28.7  │
└──────┴──────────────────┴──────┴────────┴──────────┴─────────┴───────┘

Total Salary: $49,900
Total Ownership: 224.5%
Projected Score: 287.3
```

**Analysis:**
- ❌ Players from 7 different games
- ❌ No correlation structure
- ❌ Random high-value picks
- ✅ High raw projection
- ⚠️ Low ceiling potential

---

### Enhanced Builder Output (v2.0)

```
Lineup #1 - Projection: 285.5  |  Correlation: 73.8
┌──────┬──────────────────┬──────┬────────┬──────────┬─────────┬───────┬──────┐
│ SLOT │ Player           │ Pos  │ Team   │ Opponent │ Salary  │ Proj  │ Own% │
├──────┼──────────────────┼──────┼────────┼──────────┼─────────┼───────┼──────┤
│ PG   │ Tyrese Haliburton│ PG   │ IND    │ DET      │ $9,200  │ 48.5  │ 32%  │
│ SG   │ Buddy Hield      │ SG/SF│ IND    │ DET      │ $5,400  │ 30.2  │ 12%  │ ← Stack
│ SF   │ Jayson Tatum     │ SF/PF│ BOS    │ LAL      │ $10,200 │ 52.8  │ 38%  │
│ PF   │ Jaylen Brown     │ SG/SF│ BOS    │ LAL      │ $8,800  │ 41.6  │ 28%  │ ← Stack
│ C    │ Anthony Davis    │ PF/C │ LAL    │ BOS      │ $9,900  │ 48.3  │ 35%  │ ← Bringback
│ G    │ Pascal Siakam    │ PF   │ IND    │ DET      │ $8,400  │ 39.8  │ 24%  │ ← Stack
│ F    │ LeBron James     │ SF/PF│ LAL    │ BOS      │ $9,100  │ 44.1  │ 31%  │ ← Bringback
│ UTIL │ Isaiah Stewart   │ PF/C │ DET    │ IND      │ $4,900  │ 28.7  │ 8%   │
└──────┴──────────────────┴──────┴────────┴──────────┴─────────┴───────┴──────┘

Total Salary: $49,900
Total Ownership: 208%
Projected Score: 285.5
Correlation Score: 73.8
Games: 4  |  Teams: 6

🎯 Stack Analysis:
  • IND (3x): Haliburton, Hield, Siakam
  • BOS@LAL Game Stack (4x): Tatum, Brown, Davis, LeBron
  
📊 Structure:
  • Primary Stack: IND 3-man (facing weak DET defense)
  • Game Stack: BOS@LAL (high total: 235, spread: 2.5)
  • Bring-backs: Davis + LeBron from LAL side
  • Unique exposure: IND 3-stack + BOS@LAL = 4.8% combined ownership
```

**Analysis:**
- ✅ Clear correlation structure
- ✅ Exposed to 2 high-upside games
- ✅ Lower total ownership (208% vs 224%)
- ✅ Unique stack combinations
- ✅ High ceiling if games hit
- ⚠️ Slightly lower raw projection (-1.8)
- 🎯 **Differentiated upside**

---

## Sample Lineup Summary Table

### Enhanced Builder - 20 Lineups

```
┌────────┬────────┬──────────┬────────────┬──────────┬───────┬────────┐
│ Lineup │ Proj   │ Corr     │ Total Own% │ Salary   │ Games │ Teams  │
├────────┼────────┼──────────┼────────────┼──────────┼───────┼────────┤
│   1    │ 285.5  │ 73.8     │ 208%       │ $49,900  │   4   │   6    │
│   2    │ 284.2  │ 68.2     │ 196%       │ $49,700  │   5   │   7    │
│   3    │ 283.8  │ 81.5     │ 215%       │ $49,800  │   3   │   5    │ ← Max corr
│   4    │ 283.1  │ 65.9     │ 203%       │ $49,600  │   4   │   6    │
│   5    │ 282.9  │ 77.3     │ 188%       │ $49,500  │   4   │   6    │
│   6    │ 282.4  │ 59.4     │ 221%       │ $49,900  │   5   │   7    │
│   7    │ 281.8  │ 71.2     │ 194%       │ $49,400  │   4   │   6    │
│   8    │ 281.5  │ 64.8     │ 209%       │ $49,800  │   5   │   7    │
│   9    │ 281.2  │ 75.6     │ 182%       │ $49,300  │   3   │   5    │
│  10    │ 280.9  │ 69.1     │ 198%       │ $49,700  │   4   │   6    │
│  11    │ 280.6  │ 62.3     │ 213%       │ $49,900  │   5   │   7    │
│  12    │ 280.3  │ 78.9     │ 191%       │ $49,600  │   4   │   5    │
│  13    │ 280.1  │ 66.5     │ 205%       │ $49,500  │   4   │   7    │
│  14    │ 279.8  │ 72.7     │ 197%       │ $49,400  │   4   │   6    │
│  15    │ 279.5  │ 58.2     │ 218%       │ $49,800  │   5   │   7    │
│  16    │ 279.2  │ 74.1     │ 189%       │ $49,200  │   3   │   5    │
│  17    │ 278.9  │ 67.8     │ 201%       │ $49,600  │   4   │   6    │
│  18    │ 278.6  │ 70.4     │ 195%       │ $49,500  │   4   │   6    │
│  19    │ 278.3  │ 63.1     │ 207%       │ $49,700  │   5   │   7    │
│  20    │ 278.0  │ 76.5     │ 186%       │ $49,300  │   4   │   5    │
└────────┴────────┴──────────┴────────────┴──────────┴───────┴────────┘

Average Correlation: 69.8
Average Projection: 280.9
Average Ownership: 200.8%
```

**Key Insights:**
- Correlation scores range from 58-82 (good variety)
- Higher correlation generally = lower total ownership
- Lineup #3 has max correlation (81.5) with 3 games, 5 teams
- Lineup #6 has lowest correlation (59.4) with highest ownership

---

## Real Example: Lineup #3 (Max Correlation)

```
Lineup #3 - Projection: 283.8  |  Correlation: 81.5 ⭐⭐⭐
┌──────┬──────────────────┬──────┬────────┬──────────┬─────────┬───────┬──────┐
│ SLOT │ Player           │ Pos  │ Team   │ Opponent │ Salary  │ Proj  │ Own% │
├──────┼──────────────────┼──────┼────────┼──────────┼─────────┼───────┼──────┤
│ PG   │ Damian Lillard   │ PG/SG│ MIL    │ CLE      │ $8,500  │ 42.3  │ 29%  │
│ SG   │ Giannis Antetok. │ PF/C │ MIL    │ CLE      │ $11,800 │ 62.1  │ 42%  │ ← Stack
│ SF   │ Khris Middleton  │ SF/SG│ MIL    │ CLE      │ $6,400  │ 33.8  │ 18%  │ ← Stack
│ PF   │ Donovan Mitchell │ SG/SF│ CLE    │ MIL      │ $8,900  │ 45.2  │ 35%  │ ← Bringback
│ C    │ Jarrett Allen    │ C    │ CLE    │ MIL      │ $7,200  │ 38.4  │ 22%  │ ← Bringback
│ G    │ Nikola Jokic     │ C    │ DEN    │ PHX      │ $11,500 │ 58.9  │ 44%  │
│ F    │ Devin Booker     │ PG/SG│ PHX    │ DEN      │ $9,200  │ 47.6  │ 38%  │ ← Bringback
│ UTIL │ Isaiah Stewart   │ PF/C │ DET    │ IND      │ $4,900  │ 28.7  │ 8%   │
└──────┴──────────────────┴──────┴────────┴──────────┴─────────┴───────┴──────┘

Total Salary: $49,800
Total Ownership: 215%
Projected Score: 283.8
Correlation Score: 81.5
Games: 3  |  Teams: 5

🎯 Stack Analysis:
  • MIL@CLE Full Game Stack (5x)
    - MIL: Lillard, Giannis, Middleton
    - CLE: Mitchell, Allen
    - Game Total: 236 (highest on slate)
    - Combined ownership of 5-stack: 29% × 42% × 18% × 35% × 22% = 0.97%
  
  • DEN@PHX Mini-Bringback (2x)
    - Jokic + Booker
    - Game Total: 228
    
📊 Why This Works:
  ✅ Concentrated exposure to highest total game (MIL@CLE)
  ✅ If game goes over, 5 of 8 players benefit (62.5% of lineup)
  ✅ 0.97% combined stack ownership = ultra unique
  ✅ Jokic/Booker as safety valve from DEN@PHX
  ✅ Maximum upside structure
  
⚠️ Risk Profile:
  • Very boom-or-bust
  • If MIL@CLE disappoints, lineup likely fails
  • But if it hits, massive ceiling
  • Perfect for large-field GPPs (Milly Maker)
```

---

## Correlation Score Breakdown

### How Lineup #3 Scored 81.5

```python
Base Score: 0

Game Stack Bonuses:
  MIL@CLE (5 players):
    (5-1) × game_quality × 0.1
    = 4 × 89.3 × 0.1
    = +35.7 points
    
  DEN@PHX (2 players):
    (2-1) × game_quality × 0.1  
    = 1 × 86.1 × 0.1
    = +8.6 points

Team Stack Bonuses:
  MIL (3 players):
    min(3-1, 3) × 15
    = 2 × 15
    = +30 points
    
  CLE (2 players):
    min(2-1, 3) × 15
    = 1 × 15  
    = +15 points
    
  DEN (1 player): 0 bonus
  PHX (1 player): 0 bonus

Game Diversity Penalty:
  3 games (optimal is 3-5)
  = 0 penalty

Total Correlation Score: 35.7 + 8.6 + 30 + 15 = 89.3
Normalized to 0-100 scale: 81.5
```

---

## Ownership Analysis

### Field vs Your Stacks

**Scenario:** 10,000 entry Milly Maker

```
Field Distribution (Estimated):
  BOS@LAL game stack: 35% of field
  DEN@PHX game stack: 28% of field  
  MIL@CLE game stack: 18% of field ← YOUR PICK
  Other game stacks: 19% of field

Your Edge:
  • Targeting 18% game stack vs 35% chalk stack
  • If MIL@CLE outscores BOS@LAL, massive leverage
  • Same correlation strategy, lower ownership
  
Math:
  • 35% on BOS@LAL means ~6,500 lineups have exposure
  • 18% on MIL@CLE means ~1,800 lineups (you're in 1,800)
  • If MIL@CLE wins the slate, you're competing with 1,800 not 6,500
  • Better odds to finish top 1%
```

---

## Cash Game vs GPP Comparison

### Same Slate, Different Correlation Settings

#### Cash Game Build (Correlation: 0.2)
```
Lineup - Projection: 289.1  |  Correlation: 32.4

Players spread across 7 games, 8 teams
No major stacks, focus on floor
Higher chalk ownership (245%)
Lower ceiling, higher consistency
```

#### GPP Build (Correlation: 0.8)  
```
Lineup - Projection: 283.8  |  Correlation: 81.5

Players concentrated in 3 games, 5 teams
Full game stacks + bring-backs
Lower total ownership (215%)
Lower floor, massive ceiling
```

**The Trade-off:**
- Cash sacrifices 6 points of projection to gain safety (-0.2 corr → -6 pts of variance)
- GPP sacrifices 5 points of projection to gain upside (+0.6 corr → +40 pts of ceiling)

---

## Advanced: Multi-Lineup Portfolio

### 20-Max Entry Strategy

```
Lineups 1-10: Correlation 0.70-0.75 (core strategy)
  • Mix of BOS@LAL and MIL@CLE stacks
  • Varied bring-backs
  • 200-210% total ownership

Lineups 11-15: Correlation 0.55-0.65 (balanced)
  • Lighter stacking
  • More spread across games
  • 215-225% total ownership

Lineups 16-20: Correlation 0.80-0.85 (max upside)
  • Ultra-concentrated game stacks
  • Contrarian games
  • 180-195% total ownership
  
Result: Diverse correlation exposure, multiple ways to win
```

---

## Summary: Why Enhanced is Better for GPPs

### Original Builder Lineup
- 8 random high-value players
- Projection: 287.3
- If 6 players hit, you score ~288
- If 6 players bust, you score ~180
- **Linear scaling**

### Enhanced Builder Lineup  
- 5 players from MIL@CLE + 3 fillers
- Projection: 283.8
- If MIL@CLE game hits (+20%), you score ~320
- If MIL@CLE game busts (-20%), you score ~150
- **Exponential scaling**

**The Key:** In GPPs, you need the exponential upside. The enhanced builder creates those explosive outcomes through intelligent correlation.

---

*This is what tournament-winning lineups look like!* 🏆
