// Source data for the death model.
// PRB "How many people have ever lived" benchmarks (data/prb.json) and
// OWID yearly world deaths (geo/owid_deaths.csv). Regenerate by hand if those change.

export type PrbRow = { year: number; birthsSincePrev: number; pop: number; cbr: number };

/** births in (prev benchmark, year], population at `year`, crude birth rate ‰ for the period */
export const PRB: PrbRow[] = [
  { year: -50000, birthsSincePrev: 0, pop: 2, cbr: 80 },
  { year: -8000, birthsSincePrev: 8993889771, pop: 5000000, cbr: 80 },
  { year: 1, birthsSincePrev: 46025332354, pop: 300000000, cbr: 80 },
  { year: 1200, birthsSincePrev: 26591343000, pop: 450000000, cbr: 60 },
  { year: 1650, birthsSincePrev: 12782002453, pop: 500000000, cbr: 60 },
  { year: 1750, birthsSincePrev: 3171931513, pop: 795000000, cbr: 50 },
  { year: 1850, birthsSincePrev: 4046240009, pop: 1265000000, cbr: 40 },
  { year: 1900, birthsSincePrev: 2900237856, pop: 1656000000, cbr: 40 },
  { year: 1950, birthsSincePrev: 3390198215, pop: 2499000000, cbr: 34 },
];

export const OWID_FIRST_YEAR = 1950;
/** world deaths per calendar year, OWID */
export const OWID_DEATHS: number[] = [
  48486892, 48176160, 47383364, 47239576, 46662428, 46635656, 46479064, 46880776,
  46518264, 50724510, 54612440, 49918924, 46061450, 46913210, 46822880, 48213548,
  47841524, 47568790, 47629250, 47815950, 48163468, 49384108, 47770224, 47573276,
  47478096, 47622990, 47707256, 47166790, 47293144, 46963884, 47349924, 47470790,
  47659344, 48206576, 48430624, 48772812, 48643612, 48657012, 49183184, 49178264,
  49794344, 50108430, 50281870, 50831804, 51930610, 51406890, 51410956, 51432010,
  51853504, 52183584, 52240370, 52431244, 52757536, 53077070, 53187800, 53390436,
  53281776, 53410050, 53987020, 54122624, 54268588, 54581440, 54794788, 55092620,
  55545320, 56305970, 56756910, 57572256, 57792804, 58354932, 63546316, 69728100,
  62278628, 61651610,
];
