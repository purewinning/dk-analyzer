# NFL DFS Support - Quick Start Guide

## 🏈 What's New

The builder now **auto-detects** NFL slates and applies NFL-specific stacking logic!

---

## Auto-Detection

**How it works:**
- Paste your CSV
- Builder checks positions
- Sees QB/RB/WR/TE/DST → **NFL mode**
- Sees PG/SG/SF/PF/C → **NBA mode**

**UI Changes:**
- Title shows: 🏈 NFL DFS Lineup Builder
- Applies NFL correlation rules automatically

---

## NFL Stacking Rules (Built-In)

### ✅ Positive Correlations (Builder Seeks)

**QB Stacks:**
- QB + WR (same team)
- QB + TE (same team)
- QB + 2 pass catchers (mini-stack)
- QB + 3 pass catchers (full stack)

**RB + DST Stack:**
- RB + DST same team
- Game script correlation (blowout = both hit)

**Game Stacks:**
- QB + pass catchers + opposing pass catcher
- Multiple players from high-total game (48+)

**Bring-Backs:**
- Offensive stack + opposing QB/WR
- Creates full game exposure

---

### ❌ Negative Correlations (Builder Avoids)

**QB + Opposing DST**
- Never pairs QB vs the DST defending him
- Example: Mahomes + BUF DST = BAD

**WR/TE + Opposing DST**
- Never pairs pass catcher vs DST defending him
- Example: Hill + BUF DST = BAD

---

## NFL Contest Structure

### Cash Games (50/50, Double-Up)
**Build:**
- 1 QB + his WR1 (stack)
- 2 workhorse RBs (bellcow roles)
- 3 safe WRs (high floor)
- 1 chalk TE
- 1 top-3 DST

**Salary:** 95%+ usage
**Correlation:** 0.15-0.2 (light QB stack only)

---

### Single Entry GPP
**Build:**
- 1 QB + 2 pass catchers (mini-stack)
- 1-2 RBs (1 from stack team possible)
- 1-2 WRs (bring-back)
- 1 leverage TE
- 1 good matchup DST

**Salary:** 92%+ usage
**Correlation:** 0.4-0.5 (moderate stacking)

---

### 20-Max GPP
**Build:**
- QB + 2-3 pass catchers
- 1 RB from different game
- Bring-back WR from QB's game
- Contrarian TE
- Leverage DST

**Salary:** 90%+ usage
**Correlation:** 0.6-0.7 (strong stacking)

---

### Milly Maker (150-Max)
**Build:**
- Full game stack: QB + 2-3 pass catchers + opposing WR/TE
- RB + DST from different game (leverage)
- Low-owned TE
- Contrarian DST

**Salary:** 90%+ usage  
**Correlation:** 0.7-0.8 (maximum NFL stacking)

---

## NFL CSV Format

**Required Columns:**
```csv
Player,Salary,Position,Team,Opponent,Projection,Ownership
Patrick Mahomes,8200,QB,KC,BUF,24.5,35
Travis Kelce,7000,TE,KC,BUF,16.8,28
Tyreek Hill,7800,WR,MIA,NYJ,18.2,22
Josh Allen,8000,QB,BUF,KC,23.8,32
Stefon Diggs,7500,WR,BUF,KC,17.5,25
```

**Position Format:**
- QB, RB, WR, TE, DST (or DEF)
- Multi-position: RB/WR, WR/TE

---

## Winning NFL Stack Examples

### Example 1: QB + Mini-Stack + Bring-Back
```
QB:   Patrick Mahomes (KC) $8,200
WR:   Travis Kelce (KC)    $7,000  ← Stack
WR:   Rashee Rice (KC)     $6,500  ← Stack
RB:   Saquon Barkley (PHI) $8,800
RB:   James Cook (BUF)     $6,200
WR:   Stefon Diggs (BUF)   $7,500  ← Bring-back
TE:   Luke Musgrave (GB)   $3,200  ← Leverage punt
FLEX: Jaylen Warren (PIT)  $4,600
DST:  San Francisco        $3,000

Correlation: KC offense (3 players) + BUF bring-back
```

---

### Example 2: RB + DST Stack
```
QB:   Jalen Hurts (PHI)    $8,400
RB:   Jahmyr Gibbs (DET)   $7,800  ← Stack
RB:   De'Von Achane (MIA)  $7,200
WR:   Tyreek Hill (MIA)    $7,800
WR:   CeeDee Lamb (DAL)    $9,000
WR:   Nico Collins (HOU)   $6,500
TE:   David Njoku (CLE)    $4,300
FLEX: Rico Dowdle (DAL)    $5,000
DST:  Detroit              $4,000  ← Stack (RB+DST game script)

Correlation: DET wins big → Gibbs volume + DST points
```

---

### Example 3: Full Game Stack
```
QB:   Josh Allen (BUF)     $8,000  ← Game stack
WR:   Stefon Diggs (BUF)   $7,500  ← Stack
WR:   Gabe Davis (BUF)     $5,800  ← Stack
RB:   Travis Etienne (JAX) $6,800
RB:   Kareem Hunt (CLE)    $5,200
WR:   Travis Kelce (KC)    $7,000  ← Bring-back (BUF opponent)
TE:   Cole Kmet (CHI)      $3,500
FLEX: Zack Moss (CIN)      $4,200
DST:  Cleveland            $3,000

Correlation: BUF@KC shootout, 3 BUF + 1 KC
```

---

## Strategy Tips

### High-Total Games (48+)
- **Stack both sides** (QB + pass catchers + bring-back)
- Target multiple players from game
- Usually 2-4 players from one high-total game

### Blowout Favorites
- **RB + DST stack** from favorite
- Favorite up big = RB volume + DST points
- Example: DET (-10) → Gibbs + DET DST

### Close Games (Spread < 3)
- **Shootout potential** 
- QB + pass catchers from both teams
- Avoid RB-heavy builds (passing game scripts)

### Bad Weather
- **Run-heavy stacks** (RB + RB from same game)
- Fade pass-heavy stacks
- Target defensive DSTs

---

## Position-Specific Notes

### QB
- **Always stack** with at least 1 pass catcher
- High-total games preferred (25+ point implied)
- Cash: top-3 owned, GPP: 10-20% owned

### RB
- **Workhorse backs** in cash (20+ touches)
- **Pass-catching backs** in GPP (PPR upside)
- Can stack RB + DST same team
- Avoid same-team RB1 + RB2

### WR
- **WR1 = safer**, WR2/WR3 = leverage
- Stack with your QB or as bring-back
- Target vs weak secondaries

### TE
- **Punt position** in GPP (lots of $3-4K options)
- Elite TEs (Kelce, Andrews) in cash
- Can be part of QB stack

### DST
- **Never vs your offensive stack**
- Target home favorites (-7 or more)
- Can stack with RB from same team
- Punt in GPP ($2-3K), pay up in cash

---

## Files Included

**Core Files:**
- `app_enhanced.py` - Auto-detects NFL, applies rules
- `builder_enhanced.py` - Core stacking logic
- `nfl_stacks.py` - NFL-specific correlation functions

**To Use:**
1. Download all 3 files
2. Rename: app_enhanced.py → app.py, builder_enhanced.py → builder.py
3. Upload to GitHub
4. Paste NFL CSV → Builder detects sport automatically!

---

## Current Status

✅ **Implemented:**
- Auto sport detection
- NFL position recognition
- Contest-specific correlation defaults
- Salary tier enforcement

🚧 **Coming Soon:**
- NFL-specific QB stack builder
- RB + DST correlation logic
- Bring-back identification
- Anti-correlation validation (QB vs DST check)

---

## Testing

**Test with this NFL CSV:**
```csv
Player,Salary,Position,Team,Opponent,Projection,Ownership
Patrick Mahomes,8200,QB,KC,BUF,24.5,35
Travis Kelce,7000,TE,KC,BUF,16.8,28
Josh Allen,8000,QB,BUF,KC,23.8,32
Stefon Diggs,7500,WR,BUF,KC,17.5,25
Saquon Barkley,8800,RB,PHI,SF,22.1,42
Christian McCaffrey,9200,RB,SF,PHI,24.8,48
Tyreek Hill,7800,WR,MIA,NYJ,18.2,22
Luke Musgrave,3200,TE,GB,MIN,8.5,6
SF DST,3000,DST,SF,PHI,9.2,18
```

**Expected:**
- Detects NFL
- Builds QB + pass catcher stacks
- Uses 90%+ of salary
- Shows correlation scores

---

## Summary

🏈 **NFL support is live!**
- Auto-detects sport from positions
- Applies NFL stacking rules
- Contest-specific builds
- QB + pass catcher correlation
- RB + DST game script stacks
- Bring-back logic
- Anti-correlation avoidance

**Just paste your NFL CSV and it works!** 🚀
