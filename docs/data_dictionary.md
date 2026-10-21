# Data dictionary and provenance

**Provenance: 100% synthetic.** All data are generated locally by `src/resourcemind/data/generate.py`
from a fixed random seed (default 42). No real emergency data, real places or real people are
represented. Zone names are invented; coordinates are a schematic km grid (latitude/longitude columns are
`km / 111` around (0, 0) and do **not** describe a real location). No network access is needed.

Regenerate the CSVs with `python scripts/generate_data.py` (written to `data/generated/`). The app
generates the same data in memory, so the CSVs are for inspection and reuse.

## `zones.csv` (one row per zone)

| Column | Type | Unit | Description |
|---|---|---|---|
| `zone_id` | str | – | `Z01` … `Z10` |
| `zone_name` | str | – | Invented name |
| `x_km`, `y_km` | float | km | Schematic position on a 32 × 24 km grid |
| `latitude`, `longitude` | float | synthetic degrees | `y_km / 111`, `x_km / 111` — not a real place |
| `river_side` | str | – | `west`/`east` of the schematic river at x = 16 km (used by the travel-disruption scenario) |
| `population` | int | people | Synthetic, 25,000–140,000 |
| `vulnerability_index` | float | 0–1 | Synthetic area-level proxy (Beta(2, 3)); **not** derived from any protected characteristic |
| `priority_tier` | str | – | `high` (≥ 0.55), `medium` (≥ 0.35), `standard` |
| `priority_weight` | float | – | Default policy $w = 1 + $ vulnerability_index |
| `base_daily_demand` | float | requests/day | Mean demand level used by the generator |
| `current_teams` | int | teams | Legacy allocation proportional to **population** (not demand), ≥ 1 per zone |
| `staging_capacity_teams` | int | teams | Max teams that can be based in the zone |
| `max_useful_service_per_day` | int | requests/day | Beyond this extra teams add no service (K = this × H) |

## `demand_history.csv` (one row per zone per day; 240 days × 10 zones)

| Column | Type | Unit | Description |
|---|---|---|---|
| `date` | date | – | 2025-01-06 … 2025-09-02 (synthetic calendar) |
| `zone_id` | str | – | Foreign key to zones |
| `demand_medical`, `demand_fire`, `demand_other` | int | requests | Poisson counts per category |
| `demand_total` | int | requests | Sum of the categories (validated) |

Generator: $\lambda_{it} = \text{base}_i \cdot \text{weekly}_i(\text{dow}) \cdot \text{annual}_i(t) \cdot \text{trend}_i(t) \cdot \text{spike}_{it} \cdot \text{citywide}_t$,
with weekend peaks, a ±5–12% annual cycle, 0–8% growth, zone spikes (2% of days, ×1.6–2.4) and
rare city-wide events (1.2% of days, ×1.3–1.5). Category counts are Poisson with zone-specific shares.

## `travel_minutes.csv` (square matrix)

Row = origin, column = destination. Minutes $= 4 + 1.25 \cdot d_{km} / 35\text{ km/h} \cdot 60$, rounded;
diagonal 0. Routes above the maximum repositioning time (default 45 min) are ineligible.

## `metadata.json`

Seed, sizes, start date, units and the provenance string.

## Validation (`resourcemind.data.validate`)

Required columns and dtypes, missing values, negative values, duplicate `(date, zone_id)` keys, unknown or
missing zones, equal observation counts per zone, categories summing to totals, vulnerability in [0, 1],
positive weights, square travel matrix with zero diagonal. Scenario inputs are re-validated on every change.

## Scenario data layers

| Layer | Where | Modified by |
|---|---|---|
| Historical demand | `Dataset.history` | never (read-only) |
| Scenario history | `ScenarioInputs.history` (a copy) | presets such as the surge (last 7 days × 1.9 in 3 zones) |
| Forecast demand | `ZoneForecast` | model output from the scenario history |
| Planning demand | `planning_demand()` | forecast (point or upper interval) × explicit stress multipliers |
