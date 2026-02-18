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
### Initial math exploration:
  * Extracting year->graves count mapping from available sources and models
  * Tuning the size of the Ancient Circle (all ~8 Billion graves before 8 000 B.C.E) that will be a central space, will have no hyperbolic geometry, flat space, but also will not have "time linear with distance from center" requirement as the outer part for the graveyard.
  * Exploring math for translating Circumference(radius) function into a Curvature(radius) function for radially-symmetrical 2D surface.
  * Exploring chunking algorithm only for outer part (post-ancient)

### Basic deployment stack
Dockerized React+Vite application with production config using nginx, intended to be deployed using ansible on my server utilizing virtual hosts and nginx-proxy setup.
Plotly used for visualizations related to math exploration.
React app with sliders for controlling various variables of the simulation (radius of the ancient circle, chunk size, max graves per chunk etc.) and observing the resulting graveyard plan and statistics
A streamlit (Python) app for exploring the advanced math quickly.

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
# About the author
[janczechowski.com](https://janczechowski.com)




    