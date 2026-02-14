# Hyper-Geometry Graveyard

## Concept

A 2D top-down visualization where distance from the center represents time since
the dawn of humanity. Every human who ever lived gets a grave — roughly 100 billion
of them.

**Core mapping:**

- **Center** = Dawn of humanity (~300,000 years ago)
- **Distance from center (r)** = Time elapsed since then
- **Circumference at distance r** = Available geometric space for graves
- **Number of graves at r** = How many people died that year

**The fundamental tension:** Available space grows linearly with distance (2πr),
but deaths per year grew super-linearly (roughly exponentially over the long arc
of history). This means grave *density* must increase dramatically as you move
outward from the center.

**The visual trick:** A dynamic zoom compensates for density — the viewport
shrinks in world-space as the viewer walks outward, so locally you always see
graves spaced about 1 cm apart. The result is a walk through an impossibly vast
graveyard that *feels* navigable.

## Current State: Density Explorer

Before building the full walking simulator, this prototype explores the core
mathematical relationship.

Given N points distributed in a disk where the radial count grows exponentially
(`f(r) ∝ e^(αr)`), three charts show the key quantities:

| Chart | What it shows | Shape |
|-------|--------------|-------|
| **Points per ring** | Count of graves in ring `[r, r+dr]` | Exponential |
| **Ring area** | Geometric area `π((r+dr)² − r²) ≈ 2πr·dr` | Linear |
| **Area density** | `count / area` — how packed the graves are | Super-exponential |

### Parameters

| Slider | Controls |
|--------|----------|
| **α** (growth rate) | How much denser the outer rings are vs inner |
| **Points** | Total dots in the disk (start small: 10k) |
| **Bins** | Histogram resolution for the charts |

## Tech

- React + TypeScript + Vite
- Canvas-based rendering (no iteration over 100B graves — the viewport always
  contains a manageable number of points)
- Docker + nginx for deployment
