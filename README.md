# Synthetic Dataset Generator for Solar-Powered IoT Nodes

This module implements a complete simulation pipeline for evaluating design configurations of solar-powered IoT nodes.
It models hourly irradiance, photovoltaic generation, PMU conversion efficiency, battery characteristics, and node energy consumption over an entire year (8760 hours).

The objective is to determine which combinations of **PV panel area**, **battery capacity**, and **battery autonomy** can sustain autonomous operation under realistic environmental conditions.

---

# Overview of the Simulation Pipeline

The pipeline performed by `run_full_simulation()` consists of:

1. Loading irradiance and temperature data
2. Computing PV power output for all panel sizes
3. Computing PMU-adjusted net power and battery current
4. Simulating battery State of Charge (SoC) over the entire year
5. Evaluating viability metrics per configuration
6. Calculating an optimality score for ranking configurations

All physical and design parameters are defined in `SolarIoTConfig`.

---

# Function Documentation

## `SolarIoTConfig`
Configuration class containing all physical parameters:

- Node power consumption
- PV panel performance constants (η_STC, temperature coefficient γ, NOCT)
- PMU efficiency values
- Battery charge/discharge efficiency
- Minimum allowable SoC
- Lists of realistic panel areas and battery capacities

A custom configuration can be passed to any function; otherwise defaults are used.

---

## `build_design_space(config)`
Generates all possible combinations:

```
panel_area_m2 × battery_capacity_mAh × eta_PMU
```

This function is optional: the simulation itself implicitly generates the design space while expanding PV, PMU, and battery dimensions.
It is useful for inspection, debugging, and external optimization tasks.

---

## `load_irradiance_data(filepath)`
Loads irradiance and ambient temperature from a CSV file.
Expected columns:

- Month
- Day
- Hour
- G_h (irradiance, W/m²)
- T_amb (ambient temperature, °C)

Returns a cleaned DataFrame with standardized column names.

---

## `compute_pv_power(irr_data, config)`
For each hour and each PV panel area:

- Computes module temperature using the NOCT model
- Computes temperature-corrected efficiency
- Computes electrical PV output power

Produces a long-format DataFrame with one row per:

```
hour_index × panel_area_m2
```

---

## `compute_hourly_balance(df_pv, config)`
For each PMU efficiency:

- Computes PMU-adjusted power (`P_PMU`)
- Computes net power balance (`P_BAT = P_PMU − NODE_POWER_W`)
- Converts net power to battery current (`I_BAT_mA`)

This expands the data across:

```
hour_index × panel_area_m2 × eta_PMU
```

---

## `simulate_battery_soc(df_pv_pmu, config)`
Simulates battery SoC hour by hour for each combination of:

```
panel_area_m2 × eta_PMU × battery_capacity_mAh
```

Features:

- Integration of net current in mAh
- Separate charge/discharge efficiency
- Enforced bounds: `SOC_MIN ≤ SoC ≤ 1`

The output (`df_soc`) contains the full SoC trajectory for each configuration as well as failure events.

---

## `evaluate_viability(df_soc, config)`
Aggregates per-configuration metrics:

- Hours at minimum SoC
- Hours at full SoC
- Surplus and deficit energy (mAh)
- Net energy balance
- Mean and standard deviation of SoC
- **Longest autonomous interval** before reaching SOC_MIN
- Number of failure hours

Produces the table `summary`.

---

## `compute_optimal_score(summary, w_batt, w_panel, w_auto)`
Computes a normalized score ∈ [0, 1] combining:

- Small battery capacity (better)
- Small panel area (better)
- High autonomy (better)

Any configuration with `failure_hours > 0` receives score = 0.
Returns `summary` sorted by score descending.

---

## `run_full_simulation(irradiance_filepath, config)`
Entry point for executing the complete pipeline.
Returns:

```
{
    'df_pv':         PV power data,
    'df_pv_pmu':     Power balance and battery current,
    'df_soc':        SoC simulation data,
    'summary':       Aggregated metrics and scores
}
```

---

# Input Data

The simulation requires a CSV file with columns:

- Month
- Day
- Hour
- G_h
- T_amb

Example: `raw-data/CR.csv`.

---

# Notes

This framework is designed to be:

- Extensible (e.g., tilt angles, variable loads, nonlinear PMUs)
- Scalable to large design spaces
- Compatible with external optimizers and machine-learning workflows

The modular structure allows replacing any component—PV model, battery model, efficiency parameters—without modifying the rest of the pipeline.

## Column definitions

### Dataset `df_pv`
| Column name       | Unit | Description |
|-------------------|------|-------------|
| `hour_index`      | —    | Sequential hour index starting at 0 (0–8759) |
| `Month`           | —    | Calendar month from input dataset |
| `Day`             | —    | Calendar day from input dataset |
| `Hour`            | —    | Hour of day (0–23) from input dataset |
| `panel_area_m2`   | m²   | Effective PV panel area for this record |
| `G_h`             | W/m² | Global horizontal irradiance |
| `T_amb`           | °C   | Ambient temperature |
| `T_mod`           | °C   | PV module temperature (NOCT model) |
| `eta_PV`          | —    | Temperature-corrected PV efficiency |
| `P_PV`            | W    | PV power before PMU |

### Dataset `df_pv_pmu`
| Column name       | Unit | Description |
|-------------------|------|-------------|
| `hour_index`      | —    | Sequential hour index |
| `panel_area_m2`   | m²   | PV panel area |
| `eta_PMU`         | —    | PMU efficiency value |
| `P_PMU`           | W    | PV power after PMU losses (`P_PV × η_PMU`) |
| `P_BAT`           | W    | Net power flowing into the battery (`P_PMU − NODE_POWER_W`) |
| `I_BAT_mA`        | mA   | Net battery current (charging/discharging) |
| `G_h`             | W/m² | Global horizontal irradiance |
| `T_amb`           | °C   | Ambient temperature |
| `T_mod`           | °C   | PV module temperature |
| `eta_PV`          | —    | PV efficiency |
| `P_PV`            | W    | PV power before PMU |

### Dataset `df_soc`
| Column name       | Unit | Description |
|-------------------|------|-------------|
| `hour_index`      | —    | Sequential hour index |
| `panel_area_m2`   | m²   | PV panel area |
| `eta_PMU`         | —    | PMU efficiency |
| `C_batt_mAh`      | mAh  | Battery nominal capacity |
| `I_BAT_mA`        | mA   | Net current into battery |
| `SoC`             | —    | State of charge (0–1) |
| `G_h`             | W/m² | Global horizontal irradiance |
| `T_amb`           | °C   | Ambient temperature |
| `T_mod`           | °C   | PV module temperature |
| `eta_PV`          | —    | PV efficiency |
| `P_PV`            | W    | PV power before PMU |
| `P_PMU`           | W    | Power after PMU |
| `P_BAT`           | W    | Net battery power |

### Dataset `summary`
| Column name            | Unit | Description |
|------------------------|------|-------------|
| `panel_area_m2`        | m²   | PV panel area |
| `C_batt_mAh`           | mAh  | Battery capacity |
| `eta_PMU`              | —    | PMU efficiency |
| `hours_total`          | h    | Total simulated hours |
| `hours_soc_min`        | h    | Hours at or below SOC_MIN |
| `hours_soc_full`       | h    | Hours at full charge |
| `soc_mean`             | —    | Mean state of charge |
| `soc_std`              | —    | Standard deviation of SoC |
| `surplus_mAh`          | mAh  | Total surplus current-hours |
| `deficit_mAh`          | mAh  | Total deficit current-hours |
| `net_mAh`              | mAh  | Surplus minus deficit |
| `autonomy_hours`       | h    | Longest continuous run above SOC_MIN |
| `soc_min_fraction`     | —    | Fraction of time at minimum SoC |
| `soc_full_fraction`    | —    | Fraction of time at full charge |
