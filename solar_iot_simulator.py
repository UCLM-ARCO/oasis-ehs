"""
Synthetic Dataset Generator for Solar-Powered IoT Nodes

This module contains all the logic for simulating solar-powered IoT node configurations.
It includes functions for:
- Setting up system constants and design space
- Loading and processing irradiance data
- Computing PV power output
- Computing hourly power balance
- Simulating battery State of Charge (SoC)
- Evaluating configuration viability and computing scores
"""

import numpy as np
import pandas as pd
import itertools


class SolarIoTConfig:
    """Configuration constants for the solar IoT simulator."""

    # Node parameters
    NODE_POWER_W = 0.05  # Constant node power consumption (50 mW)

    # Photovoltaic panel constants
    ETA_STC = 0.175        # Efficiency at STC
    GAMMA_PER_C = -0.0045  # Temperature coefficient (%/°C)
    NOCT_C = 45.0          # Nominal Operating Cell Temperature (°C)

    # Battery constants
    BATTERY_ETA_C = 0.95  # Charge/discharge efficiency
    SOC_MIN = 0.2         # Minimum allowed SoC (fraction)
    BATTERY_VOLTAGE = 3.7  # Nominal voltage (V)

    # Variable parameter ranges
    PMU_ETA_VALUES = [0.87, 0.90, 0.95, 0.98]

    # Photovoltaic panel areas (1–400 cm²)
    PANEL_AREAS_M2 = [
        0.0001,   # 1 cm²
        0.00025,  # 2.5 cm²
        0.0004,   # 4 cm²
        0.000625, # 6.25 cm² (≈2.5×2.5 cm)
        0.0010,   # 10 cm²
        0.0025,   # 25 cm²
        0.0040,   # 40 cm²
        0.00625,  # 62.5 cm² (≈8×8 cm cell)
        0.0080,   # 80 cm²
        0.0100,   # 100 cm² (≈10×10 cm)
        0.0160,   # 160 cm² (≈12.6×12.6 cm)
        0.0250,   # 250 cm² (≈15.8×15.8 cm)
        0.0310,   # 310 cm² (≈17.6×17.6 cm)
        0.0400    # 400 cm² (≈20×20 cm)
    ]

    # Battery capacities (realistic IoT battery sizes)
    BATTERY_CAPACITIES_MAH = [
        30,    # small Li-ion coin cell (~0.1 Wh @ 3.7 V)
        70,    # supercap or very small LiPo (~0.25 Wh @ 3.7 V)
        135,   # compact LiPo (~0.5 Wh @ 3.7 V)
        270,   # small pouch cell (~1.0 Wh @ 3.7 V)
        500,   # standard Li-ion (~2.0 Wh @ 3.7 V)
        1000,  # one 18650 cell (~3.7 Wh)
        1300,  # small LiPo pack (~5.0 Wh @ 3.7 V)
        2000,  # two small Li-ion cells (~7.4 Wh @ 3.7 V)
        2600,  # high-capacity 18650 (~10 Wh)
        4000,  # larger LiPo pack (~15 Wh)
        5400   # large IoT/multi-day autonomy pack (~20 Wh)
    ]


def build_design_space(config=None):
    """
    Generate all possible combinations of panel area, battery capacity, and PMU efficiency.

    Parameters
    ----------
    config : SolarIoTConfig, optional
        Configuration object. If None, uses default SolarIoTConfig.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: panel_area_m2, battery_capacity_mAh, eta_PMU
    """
    if config is None:
        config = SolarIoTConfig()

    design_space = list(itertools.product(
        config.PANEL_AREAS_M2,
        config.BATTERY_CAPACITIES_MAH,
        config.PMU_ETA_VALUES
    ))

    df_design = pd.DataFrame(design_space, columns=[
        "panel_area_m2",
        "battery_capacity_mAh",
        "eta_PMU"
    ])

    return df_design


def load_irradiance_data(filepath="raw-data/CR.csv"):
    """
    Load irradiance and temperature data from CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing irradiance data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Month, Day, Hour, G_h, T_amb
    """
    irr_data = pd.read_csv(filepath)

    # Basic sanity checks
    expected_cols = ["Date-hour", "Month", "Day", "Hour", "G(h)", "Temperature"]
    missing_cols = [c for c in expected_cols if c not in irr_data.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Rename columns for internal consistency
    irr_data = irr_data.rename(columns={
        "Date-hour": "datetime_str",
        "G(h)": "G_h",
        "Temperature": "T_amb"
    })

    # Keep only relevant numeric columns
    irr_data = irr_data[["Month", "Day", "Hour", "G_h", "T_amb"]].reset_index(drop=True)

    return irr_data


def compute_pv_power(irr_data, config=None):
    """
    Compute PV power output for all panel areas and hours.

    Parameters
    ----------
    irr_data : pd.DataFrame
        Irradiance data from load_irradiance_data()
    config : SolarIoTConfig, optional
        Configuration object. If None, uses default SolarIoTConfig.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per hour and panel area.
        Columns: hour_index, Month, Day, Hour, panel_area_m2, G_h, T_amb, T_mod, eta_PV, P_PV
    """
    if config is None:
        config = SolarIoTConfig()

    # Build a long-format DataFrame: one row per hour and panel area
    df_pv = pd.concat(
        [irr_data.assign(panel_area_m2=A) for A in config.PANEL_AREAS_M2],
        ignore_index=True
    )

    # Compute module temperature based on NOCT model
    df_pv["T_mod"] = df_pv["T_amb"] + (config.NOCT_C - 20.0) / 800.0 * df_pv["G_h"]

    # Compute temperature-corrected panel efficiency
    df_pv["eta_PV"] = config.ETA_STC * (1.0 + config.GAMMA_PER_C * (df_pv["T_mod"] - 25.0))

    # Compute PV power output
    df_pv["P_PV"] = df_pv["G_h"] * df_pv["panel_area_m2"] * df_pv["eta_PV"]

    # Add hour index (0..n-1)
    df_pv["hour_index"] = df_pv.groupby("panel_area_m2").cumcount()

    # Keep relevant columns
    df_pv = df_pv[[
        "hour_index",
        "Month",
        "Day",
        "Hour",
        "panel_area_m2",
        "G_h",
        "T_amb",
        "T_mod",
        "eta_PV",
        "P_PV"
    ]]

    return df_pv


def compute_hourly_balance(df_pv, config=None):
    """
    Compute hourly power balance considering PMU efficiency and node consumption.

    Parameters
    ----------
    df_pv : pd.DataFrame
        PV power data from compute_pv_power()
    config : SolarIoTConfig, optional
        Configuration object. If None, uses default SolarIoTConfig.

    Returns
    -------
    pd.DataFrame
        DataFrame with PMU-adjusted power and battery current.
        Additional columns: eta_PMU, P_PMU, P_BAT, I_BAT_mA
    """
    if config is None:
        config = SolarIoTConfig()

    # Expand df_pv for all PMU efficiencies
    df_pv_pmu = pd.concat(
        [df_pv.assign(eta_PMU=eta) for eta in config.PMU_ETA_VALUES],
        ignore_index=True
    )

    # Compute PMU-adjusted power
    df_pv_pmu["P_PMU"] = df_pv_pmu["P_PV"] * df_pv_pmu["eta_PMU"]

    # Net power (W)
    df_pv_pmu["P_BAT"] = df_pv_pmu["P_PMU"] - config.NODE_POWER_W

    # Convert net power to net current (mA)
    df_pv_pmu["I_BAT_mA"] = (df_pv_pmu["P_BAT"] / config.BATTERY_VOLTAGE) * 1000

    return df_pv_pmu


def simulate_battery_soc(df_pv_pmu, config=None):
    """
    Simulate battery State of Charge hour by hour for all configurations.

    Parameters
    ----------
    df_pv_pmu : pd.DataFrame
        Power balance data from compute_hourly_balance()
    config : SolarIoTConfig, optional
        Configuration object. If None, uses default SolarIoTConfig.

    Returns
    -------
    pd.DataFrame
        DataFrame with simulated SoC for each configuration.
        Additional columns: C_batt_mAh, SoC, failure_hour
    """
    if config is None:
        config = SolarIoTConfig()

    # Expand df_pv_pmu for all battery capacities
    df_soc = pd.concat(
        [df_pv_pmu.assign(C_batt_mAh=Cbat) for Cbat in config.BATTERY_CAPACITIES_MAH],
        ignore_index=True
    )

    df_soc["SoC"] = np.nan

    # Simulate SoC for each configuration
    # One configuration = (panel_area_m2, C_batt_mAh, eta_PMU)
    for (A, Cbat, eta), group_idx in df_soc.groupby(
        ["panel_area_m2", "C_batt_mAh", "eta_PMU"]
    ).groups.items():

        idx = list(group_idx)
        i_bat = df_soc.loc[idx, "I_BAT_mA"].to_numpy()

        soc = np.empty_like(i_bat)
        soc[0] = 1.0  # start fully charged

        for i in range(1, len(i_bat)):
            delta = i_bat[i]

            if delta >= 0:
                # Charging: apply charge efficiency
                soc[i] = soc[i-1] + (delta / Cbat) * config.BATTERY_ETA_C
            else:
                # Discharging: apply discharge efficiency
                soc[i] = soc[i-1] + (delta / Cbat) / config.BATTERY_ETA_C

            # Clamp SoC to allowed range
            soc[i] = min(1.0, max(config.SOC_MIN, soc[i]))

        df_soc.loc[idx, "SoC"] = soc

    # Compute failure hours at the df_soc level
    # A "failure hour" is when:
    #   (1) SoC == SOC_MIN  → battery cannot discharge further
    #   (2) I_BAT_mA < 0     → the node still requires power
    df_soc["failure_hour"] = (
        (df_soc["SoC"] <= config.SOC_MIN + 1e-9) &
        (df_soc["I_BAT_mA"] < 0)
    ).astype(int)

    return df_soc


def longest_autonomy_hours(soc_series, soc_min):
    """
    Compute the longest continuous interval (in hours) during which
    SoC stays strictly above soc_min.

    Parameters
    ----------
    soc_series : array-like
        Series of SoC values
    soc_min : float
        Minimum SoC threshold

    Returns
    -------
    int
        Longest continuous autonomy in hours
    """
    below = soc_series <= soc_min + 1e-6
    if np.all(below):
        return 0

    max_len = 0
    current = 0
    for v in below:
        if not v:
            current += 1
            max_len = max(max_len, current)
        else:
            current = 0
    return max_len


def evaluate_viability(df_soc, config=None):
    """
    Compute aggregated performance metrics per configuration.

    Parameters
    ----------
    df_soc : pd.DataFrame
        SoC simulation data from simulate_battery_soc()
    config : SolarIoTConfig, optional
        Configuration object. If None, uses default SolarIoTConfig.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with metrics per configuration.
    """
    if config is None:
        config = SolarIoTConfig()

    summary = (
        df_soc
        .groupby(["panel_area_m2", "C_batt_mAh", "eta_PMU"], as_index=False)
        .agg(
            hours_total=("SoC", "count"),
            hours_soc_min=("SoC", lambda s: np.sum(s <= config.SOC_MIN + 1e-6)),
            hours_soc_full=("SoC", lambda s: np.sum(s >= 1.0 - 1e-6)),
            soc_mean=("SoC", "mean"),
            soc_std=("SoC", "std"),
            surplus_mAh=("I_BAT_mA", lambda s: np.sum(np.clip(s, 0, None))),
            deficit_mAh=("I_BAT_mA", lambda s: -np.sum(np.clip(s, None, 0))),
            autonomy_hours=("SoC", lambda s: longest_autonomy_hours(s.to_numpy(), config.SOC_MIN)),
            failure_hours=("failure_hour", "sum")
        )
    )

    # Derived metrics
    summary["soc_min_fraction"] = summary["hours_soc_min"] / summary["hours_total"]
    summary["soc_full_fraction"] = summary["hours_soc_full"] / summary["hours_total"]
    summary["net_mAh"] = summary["surplus_mAh"] - summary["deficit_mAh"]

    return summary


def compute_optimal_score(summary, w_batt=1.0, w_panel=1.0, w_auto=1.0):
    """
    Compute an optimality score for each configuration.

    Score is between 0 and 1, where:
    - 1 = best possible configuration
    - 0 = worst possible configuration

    Any configuration with failure_hours > 0 receives score = 0.

    Parameters
    ----------
    summary : pd.DataFrame
        Summary DataFrame from evaluate_viability()
    w_batt : float
        Weight for battery capacity (smaller is better)
    w_panel : float
        Weight for panel area (smaller is better)
    w_auto : float
        Weight for autonomy (higher is better)

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with added 'score' column, sorted by score descending.
    """
    def compute_raw_score(row):
        """Compute unnormalized raw score."""
        # Normalize battery capacity (smaller = better → invert)
        batt_norm = (row["C_batt_mAh"] - summary["C_batt_mAh"].min()) / \
                    (summary["C_batt_mAh"].max() - summary["C_batt_mAh"].min())
        batt_score = 1.0 - batt_norm

        # Normalize panel area (smaller = better → invert)
        panel_norm = (row["panel_area_m2"] - summary["panel_area_m2"].min()) / \
                     (summary["panel_area_m2"].max() - summary["panel_area_m2"].min())
        panel_score = 1.0 - panel_norm

        # Normalize autonomy (higher = better)
        auto_norm = (row["autonomy_hours"] - summary["autonomy_hours"].min()) / \
                    (summary["autonomy_hours"].max() - summary["autonomy_hours"].min())

        # Weighted raw score
        raw = (
            w_batt * batt_score +
            w_panel * panel_score +
            w_auto * auto_norm
        )

        return raw

    # Compute raw scores for all rows
    summary["raw_score"] = summary.apply(compute_raw_score, axis=1)

    # Normalize raw_score to [0,1]
    raw_min = summary["raw_score"].min()
    raw_max = summary["raw_score"].max()

    summary["score"] = (summary["raw_score"] - raw_min) / (raw_max - raw_min + 1e-12)

    # Apply strict penalty for failures
    # Any configuration with failure_hours > 0 gets score = 0
    summary.loc[summary["failure_hours"] > 0, "score"] = 0.0

    del summary["raw_score"]

    # Sort table by score (best first)
    summary = summary.sort_values("score", ascending=False).reset_index(drop=True)

    return summary


def run_full_simulation(irradiance_filepath="raw-data/CR.csv", config=None):
    """
    Run the complete simulation pipeline.

    Parameters
    ----------
    irradiance_filepath : str
        Path to the irradiance data CSV file
    config : SolarIoTConfig, optional
        Configuration object. If None, uses default SolarIoTConfig.

    Returns
    -------
    dict
        Dictionary containing:
        - 'df_pv': PV power data
        - 'df_pv_pmu': Power balance data
        - 'df_soc': SoC simulation data
        - 'summary': Summary metrics with scores
    """
    if config is None:
        config = SolarIoTConfig()

    print("Loading irradiance data...")
    irr_data = load_irradiance_data(irradiance_filepath)

    print("Computing PV power...")
    df_pv = compute_pv_power(irr_data, config)

    print("Computing hourly balance...")
    df_pv_pmu = compute_hourly_balance(df_pv, config)

    print("Simulating battery SoC...")
    df_soc = simulate_battery_soc(df_pv_pmu, config)

    print("Evaluating viability...")
    summary = evaluate_viability(df_soc, config)

    print("Computing optimal scores...")
    summary = compute_optimal_score(summary)

    print("Done!")

    return {
        'df_pv': df_pv,
        'df_pv_pmu': df_pv_pmu,
        'df_soc': df_soc,
        'summary': summary
    }
