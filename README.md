# Untitled Graveyard Project
A 2D exploration space containing a grave for each individual human who ever lived.
Based on modern statistics and models of historical and ancient population this means over 100 billion individual graves.
# The map of the graveyard
The graveyard has a circular plan, the center of it represents the dawn of humanity.
Apart from an "ancient circle" in the middle representing all graves before 8000 B.C.E., the time is intended to increase *linearly* as you move away from the center.
The very edge of the graveyard represents people dying right now, in the 21st century.
## Hyperbolic (non-Euclidean) geometry
Project tries to achieve the following:
* Maintain roughly uniform grave density across the graveyard — avoiding areas that are either sparsely populated or overcrowded
* Explosive population (and deaths) growth in modern era
* Time moving forward with constant speed as you move towards the edge (apart from the ancient circle in the middle)
But the important complication is:
- On circular graveyard plan, available space for graves (circumference) only grows linearly with the distance from the center
- But the space we need grows super-linearly
The solution is:
- Use non-Euclidean geometry. The space will look flat only locally but will be curved in a complex way. After you go the distance R from the center, getting close to the edge, going all the way around the graveyard will take you much more than 2π*R
It is mathematically possible to accommodate for super-linearly growing circumference by utilizing *negative Gaussian curvature*.
Which in our case will be changing with the distance from the center depending on how many graves we need to 'fit' for a given year.
### Mental model of how hyperbolic surface will be used in this project
Imagine a sphere - the Earth.
You stand on the north pole and draw a circle on the ground with range R centered right at the pole - the circumference is 2π*R.
But what if you make the circle bigger and bigger, so big it is as big as the equator?
The radius of your circle (as you measure it on the surface) is the distance from the north pole to the equator - roughly 10 000 km.
But the circumference of the earth (or, the length of the equator) is roughly 40 000 km - much less than 2π * 10 000 km.
This is a very easy to imagine example of surface that looks locally flat (hard to say that the Earth is curved from the perspective of an ant) but where circumference of a circle grows sub-linearly with the radius.
It is possible to achieve the same in reverse.
Such surfaces are called "Hyperbolic" surfaces.
However such surfaces don't have a 3D, physical interpretation (like the sphere in Earth's case) so they are difficult to visualize.
But they can be simulated and explored.
And if your view port is very small in comparison to the curvature of the space it will look very normal on your screen - things will only get "weird" once you measure distances over long paths.

## Project status
### Walkable prototype (`/`, code in `frontend/src/graveyard/`)
A 2D walkable graveyard with geometry fixed by the historical data. Everything is smooth by construction: no creases, no curvature jumps.
* **Death model** (`deathModel.ts`), 50 000 BCE → 2030, D = P·m with both factors quintic B-splines (so D is C⁴):
  * P: smooth log-space fit of PRB's own exponential-per-period estimates and OWID decade means (positive by construction);
  * m ≈ 1: smoothest multiplier (min ∫m′² + α·m‴²) that makes every period total exact: all PRB periods including the single pre-8000 BCE total, and each OWID decade.
  * Both steps are single linear (KKT) solves: no iterative fitting, nothing to converge.
* **Surface** (`surface.ts`): metric ds² = dρ² + f(ρ)²dφ², uniform grave density σ (one plot = 2.6 m × 1.3 m).
  * **History** (from the switch year, default 3000 BCE): time is linear (v m per year), and uniform density forces f = D(t)/(2πσv), so curvature is K = −D″/(D·v²). Year markers start here.
  * **Ancient era** (`ancient.ts`): one undated bucket for everyone who died before the switch; its length is set as a **share of the walk** (default 10%). No pieces: one smooth curve f from the centre (flat there, f ≈ ρ) to the rim, matching the history ring and its first three derivatives there (K is C¹ across) and holding exactly the graves before the switch. The unknown is the ring growth rate f′ as a spline; every constraint (centre, edge, area) is linear, so the smoothest curve (min ∫x²(f‴/f)²) is one linear solve per reweighting pass. Rings may grow past the history ring and narrow back at the rim (a sphere-like bulb): that is what lets the ancient era be short. The price is curvature, reported against the view width.
  * D is C⁴ and the ancient curve joins history C³, so curvature is C² everywhere except C¹ at the rim: no creases.
  * **Why the bulb**: if the ancient rings could not outgrow the history ring, the ancient era would need at least N_before·v/D(switch) (≈2 570 yr for a 3000 BCE switch, a third of the walk). Letting them bulge removes that floor; a 10% ancient era peaks at rings ~20× the history ring, tightest curvature radius ~15 yr.
  * **Flatten after**: the death rate right after the switch starts at the window mean and rises more slowly, with the same PRB totals (PRB only fixes the 8000 BCE → 1 CE total, so the shape inside is free). The window may not cross the next PRB benchmark. Default: flatten 3000 BCE → 0, which raises D(switch) from 5.7 to 8.0 M/yr and lowers the floor from 41% to 34%.
  * **Default** (preset *Short walk*: 3000 BCE switch, full flatten, ancient 10%): **559 yr ancient + 5 030 yr history** in ≈20 min at 4.66 yr/s. Density 0.22 /yr² (0.43 rows per yr) so you pass 2 graves/s; view 80 yr, a screen every 17 s.
  * Rendering uses the conformal chart u = ∫dρ/f: screen = f(ρ_player)·(e^(Δu+iΔφ) − 1), exact at the player.
* **Units**: the interface measures distance in history-years (1 yr = the radial depth of one year of history). Uniformly rescaling every length gives the same world, so in years there is one density knob: **grave density** in graves per yr² (σv²). Plots are fixed 1.2 m squares, so density also fixes metres per year (v = √(density/σ)). The experience is set by three numbers: density, **speed** (yr/s) and **view width** (yr). History curvature in 1/yr² is −D″/D: data only.
* **Layout** (`grid.ts`): square 1.2 m plots in back-to-back pairs of rows, so every grave touches a narrow ring path. 5 pairs make a block, blocks are separated by ring roads, every 6th ring road is an avenue. Radial roads branch in two where the ring doubles (per block), every 8th is an avenue, and narrow radial cut-throughs split each segment into runs aligned through the block. Grave counts per row come from uniform density over the whole cycle (paths included), so ids stay in death order. A grave knows only its id and (from its place in that order) an approximate year of death; nothing else is invented. The newest row fills in real time.
* **Stones** (`sprites.ts`): generic memorials with no religious or culture-specific symbols and no claim about what graves looked like in any era: six shapes (rounded, flat, gabled, low block, obelisk, rough stone) picked by hash, engraved lines as the inscription. Only their weathering changes with age (ancient era, older than ~250 years, recent). One sprite per grave: the monument inside its square; small graves become simple blocks, and far away the ground is one tinted fill.
* **Walking**: fixed speed in yr/s (set by the preset) or explore speed (fraction of view per second); collisions keep you on paths (toggle for debugging). Jumps and map travel land on the nearest path. Optional procedural soundscape (`soundscape.ts`: wind, drone deepening in the ancient era, crickets, owl, gravel footsteps).
* **Settings dialog**: presets defined by walk time, graves passed per second and seconds per screen, from which density, speed and view width follow (*Short walk* ≈20 min, *Quick look* ≈8 min, *Long walk* ≈1 h, *Explore* for debugging); an experience panel (walk times, graves passed per second, graves along one radius, time to cross the screen, graves on screen, plot size); three main settings: ancient share, pace (graves/s, sets the speed) and view width; the curvature report and live ring-length / curvature charts. Density, history start, flatten window, fit settings and Shift boost are under *advanced*. World changes are fitted and applied straight away (restart at the centre). Settings persist in localStorage. The **model & charts** dialog plots D(t) against the sources and proves every period total by direct integration. `npm run model-report` prints the fit and geometry.

### Initial math exploration:
  * Extracting year->graves count mapping from available sources and models
  * Tuning the size of the Ancient Circle (all ~8 Billion graves before 8 000 B.C.E) that will be a central space, will have no hyperbolic geometry, flat space, but also will not have "time linear with distance from center" requirement as the outer part for the graveyard.
  * Exploring math for translating Circumference(radius) function into a Curvature(radius) function for radially-symmetrical 2D surface.
  * Exploring chunking algorithm only for outer part (post-ancient)

### Basic deployment stack
Dockerized React+Vite application with production config using nginx, intended to be deployed using ansible on my server utilizing virtual hosts and nginx-proxy setup.
Plotly used for visualizations related to math exploration.
React app with sliders for controlling various variables of the simulation (radius of the ancient circle, chunk size, max graves per chunk etc.) and observing the resulting graveyard plan and statistics

## Next steps
* Experiment with curved ancient circle - might fit more graves on smaller radius without compromising the (global!) density
* Basic time math - time to get to the edge depending on speed / number of graves passed per second
* Trajectory simulations, mostly to double check the math: spawn at point (r, f) go df for n steps, plot the trajectory on radial. how quickly to you get back. What if you go to (r,f) and then continue in a geodesic in arbitrary direction?
* Update chunk math for curved space, also add time calculations (number of chunks until the edge, chunks loaded per second)
* View port basic math with size sliders (number of chunks in view port, number of graves in view port, curvature change between edges)
* View port experiments - toy layout with constant curvature and grid points. what do you see at various points? snapshots of moving away from the center. animation of moving away from the center, interactive "hover over map to render view port below", interactive "press arrows to move view port"

### View port ideas
* Circle? 2.5d? Rotation: always edge-ward /center-ward on top?
* Is view port constant area? Or edges are line-fragments from the surface in right distance? Or collection of graves with distance less than X. Shape of the view port on surface?

## Minimal prototype and technical requirements (not completed)
A 2D explorable space you can open in your browser, walk around using your keyboard.
You can get from the center (your starting point) to the edge of the graveyard within 1h.
You see graves rendered on your screen on their positions top-down, with your position in the center of the screen.
You see less than 1000 graves at your screen at once.
Traversing the length of your screen takes you at least a couple of seconds (graves are not flashing behind your eyes as you press down on an arrow to move)
Grave count and its relation to distance is fixed, but exact positions of the graves are random / procedurally generated. They can change between runs.
Browser memory does not need to store more than 100 000 graves at once.
It also doesn't need to generate more than 200 000 grave positions per second when player moves in constant direction.
Exact positions of 100B graves are not stored in any DB, they are generated on the fly as player explores the space

## Long term features to consider
* Map view showing your position in the graveyard (HUD minimap or a dialog)
* (to be decided, might break the experience) ability to quickly move to a different part of the graveyard based on the map
* "Signs" spawned at specific distances from the center giving you information of your current location and trivia about the graveyard and related numbers
* Something to give you a sense of passage of time, perhaps lines showing specific years, or current year in the HUD
* Radial slices of the graveyard representing different regions of the world
  * Close to the edge we can split by specific countries
  * For earlier dates we can split by continents
  * For example: North from the center is Asia, North-East is Europe, etc.
  * Regions also represented on the map
* Each grave has unique id within its chunk
  * Chunk X grave Y will always be roughly in the same position in the graveyard, the place rendered within the chunk might vary between runs (unless we make chunk grave layout generation deterministic).
  * This opens a way for grave specific features even if we don't have a database of ALL 100B of them
  * On click, see ad-hoc, procedurally generated: gender, region, year of birth (and age at death - might be more advanced modelling) - random but realistic and deterministic for specific id - the overall statistics will match the data
  * Pick specific grave ids from right period / region as graves of individuals having a Wikipedia page. Link to this page. Show the "Wikipedia" individuals on the map (they will be sparse and it might be a journey to get to the closest one)
  * On any grave, add an option to leave a flower or remembrance message. It should have some friction (so it takes time / multiple steps) but then this information is saved in the database and will be visible to other players who find this grave
  * Allow non-profits related to historical figures to link to their pages / leave a bio in a selected historical grave, perhaps for a small fee helping fund the project
* On the edge, see new graves forming in real time based on known death rates.
* The Frontier of Light - represent whole humanity here, not only the dead
  * What if instead of grave, we have candles. For dead people already put down, but for the living 9B individuals there are many alight candles at the edge. Close to edge you see some candles being alight (some people born in early 20th century are still alive) and at the edge and beyond its (almost) only light. You see candles going off based on death rates. You see new candles appear at the edge based on birth rates
* Visited by N players before / Last visited - this feature might be implementable without a huge database if we do it on the chunk level. Then players can see if they are taking a new path through the space and seeing the graves noone has seen before. Just increment count for the chunk id every time a player enters
* Maybe make surface proportional to population not deaths - this way in periods when the death rate increases(black death, wars) the density will be higher, and in modern period it will also become generally lower. I think that might be a nice visual but complicates math
# About the author
[janczechowski.com](https://janczechowski.com)




    