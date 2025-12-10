#!/usr/bin/env python

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

try:
    from numba import njit
except ModuleNotFoundError:
    def njit(func=None, **kwargs):
        # Numba unavailable -> return function unchanged
        if func is None:
            return lambda f: f
        return func


class BatterySpec:
    def __init__(self, capacity_Ah, I_max_A, kind='', description=""):
        self.capacity_Ah = capacity_Ah  # Battery capacity in Ah
        self.max_discharge_A = I_max_A  # Maximum continuous discharge current (A)
        self.kind = kind
        self.description = description

    def __repr__(self):
        return f"BatterySpec(cap={self.capacity_Ah} Ah, Imax={self.max_discharge_A} A)"


class Config:
    """Configuration constants for the solar IoT simulator."""

    # Node parameters
    NODE_POWER_W = 0.05  # Constant node power consumption (50 mW)

    # Photovoltaic panel constants
    ETA_STC = 0.175        # Efficiency at STC
    GAMMA_PER_C = -0.0045  # Temperature coefficient (%/°C)
    NOCT_C = 45.0          # Nominal Operating Cell Temperature (°C)

    # Battery constants
    BATTERY_ETA_C = 0.95   # Charge/discharge efficiency
    SOC_MIN = 0.2          # Minimum allowed SoC (fraction)
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

    BATTERY_SPECS = [
        BatterySpec(0.030, 0.10, "Li-ion",
                    "small Li-ion coin cell (~0.1 Wh)"),
        BatterySpec(0.070, 0.20, "LiPo",
                    "supercap / very small LiPo (~0.25 Wh)"),
        BatterySpec(0.135, 0.40, "LiPo",
                    "compact LiPo (~0.5 Wh)"),
        BatterySpec(0.270, 0.80, "LiPo",
                    "small pouch (~1 Wh)"),
        BatterySpec(0.500, 1.50, "Li-ion",
                    "standard Li-ion (~2 Wh)"),
        BatterySpec(1.000, 3.00, "Li-ion",
                    "single 18650 (~3.7 Wh)"),
        BatterySpec(1.300, 4.00, "LiPo",
                    "small LiPo pack (~5 Wh)"),
        BatterySpec(2.000, 6.00, "Li-ion",
                    "two Li-ion (~7.4 Wh)"),
        BatterySpec(2.600, 8.00, "Li-ion",
                    "high capacity 18650 (~10 Wh)"),
        BatterySpec(4.000, 10.0, "LiPo",
                    "large LiPo (~15 Wh)"),
        BatterySpec(5.400, 12.0, "Li-ion",
                    "multi-day pack (~20 Wh)")
    ]

    # Melisa
    # https://docs.google.com/spreadsheets/d/16a1q3lofZJNS3nguHSokKgYhDOOP0ylY17oNx9dazzg/edit?gid=0#gid=0
    BATTERY_SPECS = [
        BatterySpec(0.105, 0.1575, "LiPo",
                    "105 mAh (401230) — SHENZHEN PKCELL — https://cdn-shop.adafruit.com/product-files/2750/LP552035_350MAH_3.7V_20150906.pdf"),

        BatterySpec(0.350, 0.5250, "LiPo",
                    "350 mAh — https://cdn-shop.adafruit.com/product-files/2750/LP552035_350MAH_3.7V_20150906.pdf"),

        BatterySpec(0.420, 0.6300, "LiPo",
                    "420 mAh — https://cdn-shop.adafruit.com/product-files/4236/4236_ds_LP552535+420mAh+3.7V.pdf"),

        BatterySpec(0.500, 0.5000, "LiPo",
                    "500 mAh — https://cdn-shop.adafruit.com/product-files/1578/Datasheet.pdf"),

        BatterySpec(0.400, 0.6000, "LiPo",
                    "400 mAh — https://cdn-shop.adafruit.com/product-files/3898/3898_specsheet_LP801735_400mAh_3.7V_20161129.pdf"),

        BatterySpec(1.200, 1.2000, "LiPo",
                    "1200 mAh — https://cdn-shop.adafruit.com/product-files/258/C101-_Li-Polymer_503562_1200mAh_3.7V_with_PCM_APPROVED_8.18.pdf"),

        BatterySpec(2.000, 2.0000, "LiPo",
                    "2000 mAh — https://cdn-shop.adafruit.com/datasheets/LiIon2000mAh37V.pdf"),

        BatterySpec(2.500, 1.5000, "LiPo",
                    "2500 mAh — https://cdn-shop.adafruit.com/product-files/328/LP785060+2500mAh+3.7V+20190510.pdf"),

        BatterySpec(0.120, 0.2400, "Li-ion",
                    "120 mAh coin cell (LIR2450) — https://cdn-shop.adafruit.com/datasheets/LIR2450.pdf"),

        BatterySpec(1.600, 4.8000, "LiFePO4",
                    "1600 mAh (high-discharge) — https://www.antbatt.com/wp-content/uploads/2019/09/18650-3.2V-1600mAh-datasheet.pdf"),

        BatterySpec(3.200, 9.6000, "LiFePO4",
                    "3200 mAh (high-discharge) — https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/180/P3_2D00_Datasheet-Cell--3232-LFP-26650.pdf"),

        BatterySpec(2.300, 46.0000, "LiFePO4",
                    "2300 mAh (very high-discharge) — https://docs.rs-online.com/4ad1/0900766b812fdd10.pdf")
    ]


@njit
def simulate_soc_kernel(i_bat, Cbat, soc_min, eta_c):
    n = len(i_bat)
    soc = np.empty(n)
    soc[0] = 1.0

    for i in range(1, n):
        delta = i_bat[i]
        if delta > 0.0:
            soc[i] = soc[i-1] - (delta / Cbat) / eta_c
        elif delta < 0.0:
            soc[i] = soc[i-1] - (delta / Cbat) * eta_c
        else:
            soc[i] = soc[i-1]

        # Clamp SoC to [SOC_MIN, 1]
        if soc[i] > 1.0:
            soc[i] = 1.0
        elif soc[i] < soc_min:
            soc[i] = soc_min

    return soc



def safe_normalized_autonomy(autonomy_value, min_auto, max_auto):
    """
    Return a stable, NaN-free normalized autonomy value in [0, 1].
    - autonomy_value: row["autonomy_hours"]
    - min_auto, max_auto: summary["autonomy_hours"].min(), .max()

    Normalization:
        (autonomy_value - min_auto) / (max_auto - min_auto)

    Rules:
    - If autonomy_value is NaN → worst case (1.0).
    - If denominator is zero or invalid → return 0.0.
    - Any NaN/inf result is sanitized.
    """

    # Worst case: missing autonomy => 1.0
    if pd.isna(autonomy_value):
        return 1.0

    den = max_auto - min_auto

    # Avoid divide-by-zero or undefined domain
    if den <= 0 or pd.isna(den):
        return 0.0

    # Standard normalization
    norm = (autonomy_value - min_auto) / den

    # Stabilize numeric issues
    if pd.isna(norm) or np.isinf(norm):
        return 1.0

    # Clip just in case due to FP rounding
    return float(np.clip(norm, 0.0, 1.0))


class Simulator:
    def __init__(self, config=None):
        if config is None:
            config = Config()
        self.config = config

    def build_design_space(self):
        """
        Generate all possible combinations of panel area, battery capacity, and PMU efficiency.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: panel_area_m2, battery_capacity_mAh, eta_PMU
        """
        design_space = list(itertools.product(
            self.config.PANEL_AREAS_M2,
            [spec.capacity_Ah for spec in self.config.BATTERY_SPECS],
            self.config.PMU_ETA_VALUES
        ))

        df_design = pd.DataFrame(design_space, columns=[
            "panel_area_m2",
            "battery_capacity_Ah",
            "eta_PMU"
        ])

        return df_design

    def load_irradiance_data(self, filepath="raw-data/CR.csv"):
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

        return irr_data[["Month", "Day", "Hour", "G_h", "T_amb"]].reset_index(drop=True)

    def compute_pv_power(self, irr_data):
        """
        Compute PV power output for all panel areas and hours.

        Parameters
        ----------
        irr_data : pd.DataFrame
            Irradiance data from load_irradiance_data()

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with one row per hour and panel area.
            Columns: hour_index, Month, Day, Hour, panel_area_m2, G_h, T_amb, T_mod, eta_PV, P_PV
        """
        # Build a long-format DataFrame: one row per hour and panel area
        df_pv = pd.concat(
            [irr_data.assign(panel_area_m2=A) for A in self.config.PANEL_AREAS_M2],
            ignore_index=True
        )

        # Compute module temperature based on NOCT model
        df_pv["T_mod"] = df_pv["T_amb"] + (self.config.NOCT_C - 20.0) / 800.0 * df_pv["G_h"]

        # Compute temperature-corrected panel efficiency
        df_pv["eta_PV"] = self.config.ETA_STC * (1.0 + self.config.GAMMA_PER_C * (df_pv["T_mod"] - 25.0))

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

    def compute_hourly_balance(self, df_pv):
        """
        Compute hourly power balance considering PMU efficiency and node consumption.

        Parameters
        ----------
        df_pv : pd.DataFrame
            PV power data from compute_pv_power()

        Returns
        -------
        pd.DataFrame
            DataFrame with PMU-adjusted power and battery current.
            Additional columns: eta_PMU, I_PV_raw_A, I_PV_to_load_A, I_load_A, I_BAT_A

            Convention:
            - I_BAT_A > 0  => battery discharging
            - I_BAT_A < 0  => battery charging
        """
        # Expand df_pv for all PMU efficiencies
        df_pv_pmu = pd.concat(
            [df_pv.assign(eta_PMU=eta) for eta in self.config.PMU_ETA_VALUES],
            ignore_index=True
        )

        # PV current before PMU -> I_PV_raw_A = P_PV / Vbat
        df_pv_pmu["I_PV_raw_A"] = df_pv_pmu["P_PV"] / self.config.BATTERY_VOLTAGE

        # PV current available to the load (η_PMU applies)
        df_pv_pmu["I_PV_to_load_A"] = df_pv_pmu["I_PV_raw_A"] * df_pv_pmu["eta_PMU"]

        # Load current (constant power node)
        I_load_A = self.config.NODE_POWER_W / self.config.BATTERY_VOLTAGE
        df_pv_pmu["I_load_A"] = I_load_A

        # Net current needed from battery:
        #   net > 0 => panel insufficient => battery DISCHARGES
        #   net < 0 => panel exceeds load => surplus charges battery
        net = df_pv_pmu["I_load_A"] - df_pv_pmu["I_PV_to_load_A"]

        # Battery current with PMU efficiency:
        #   Discharge:  I_BAT_A =  net / η_PMU     (> 0)
        #   Charge:     I_BAT_A =  net * η_PMU     (< 0)
        df_pv_pmu["I_BAT_A"] = np.where(
            net >= 0.0,
            net / df_pv_pmu["eta_PMU"],
            net * df_pv_pmu["eta_PMU"]
        )

        return df_pv_pmu


        return df_pv_pmu

    def simulate_battery_soc(self, df_pv_pmu):
        """
        Simulate battery State of Charge hour by hour for all configurations.

        Parameters
        ----------
        df_pv_pmu : pd.DataFrame
            Power balance data from compute_hourly_balance()

        Returns
        -------
        pd.DataFrame
            DataFrame with simulated SoC and failure flags for each configuration.
            Columns added: C_batt_Ah, I_batt_max_A, SoC, failure_hour
        """

        # Expand dataframe for all battery specifications (capacity + I_max)
        df_soc = pd.concat(
            [
                df_pv_pmu.assign(
                    C_batt_Ah=spec.capacity_Ah,
                    I_batt_max_A=spec.max_discharge_A
                )
                for spec in self.config.BATTERY_SPECS
            ],
            ignore_index=True
        )

        df_soc["SoC"] = np.nan

        grouped = df_soc.groupby(["panel_area_m2", "C_batt_Ah", "eta_PMU"])

        for key, idx in grouped.groups.items():
            group = df_soc.loc[idx].sort_values("hour_index")
            ordered_idx = group.index.to_numpy()

            i_bat = group["I_BAT_A"].to_numpy()
            Cbat = float(key[1])

            soc = simulate_soc_kernel(
                i_bat.astype(np.float64),
                Cbat,
                float(self.config.SOC_MIN),
                float(self.config.BATTERY_ETA_C)
            )

            df_soc.loc[ordered_idx, "SoC"] = soc

        # Failure 1: SoC_min while still requiring power
        fail_soc_min = (
            (df_soc["SoC"] <= self.config.SOC_MIN + 1e-9) &
            (df_soc["I_BAT_A"] > 0)
        )

        # Failure 2: discharge current exceeds battery max current
        # Battery discharging -> I_BAT_A > 0
        fail_peak_current = (
            (df_soc["I_BAT_A"] > df_soc["I_batt_max_A"]) &
            (df_soc["I_BAT_A"] > 0)
        )

        df_soc["failure_hour_peak"] = fail_peak_current.astype(int)

        # (3) Unified failure flag
        df_soc["failure_hour"] = (fail_soc_min | fail_peak_current).astype(int)

        return df_soc

    def longest_autonomy_hours(self, soc_series, soc_min):
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

    def evaluate_viability(self, df_soc):
        """
        Compute aggregated performance metrics per configuration.

        Parameters
        ----------
        df_soc : pd.DataFrame
            SoC simulation data from simulate_battery_soc()

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with metrics per configuration.
            Additional columns:
                - I_batt_max_A   : maximum allowed discharge current for that battery
                - I_req_max_A    : maximum required discharge current (A)
        """

        def longest_autonomy_hours(s, soc_min):
            arr = s.to_numpy()
            below = arr <= soc_min + 1e-6
            if below.all():
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

        grouped = df_soc.groupby(
            ["panel_area_m2", "C_batt_Ah", "eta_PMU"],
            as_index=False
        )

        summary = grouped.agg(
            hours_total=("SoC", "count"),

            hours_soc_min=("SoC", lambda s: np.sum(s <= self.config.SOC_MIN + 1e-6)),
            hours_soc_full=("SoC", lambda s: np.sum(s >= 1.0 - 1e-6)),

            soc_mean=("SoC", "mean"),
            soc_std=("SoC", "std"),

            surplus_Ah=("I_BAT_A", lambda s: np.sum(np.clip(s, 0, None))),
            deficit_Ah=("I_BAT_A", lambda s: -np.sum(np.clip(s, None, 0))),


            autonomy_hours=("SoC", lambda s: self.longest_autonomy_hours(s, self.config.SOC_MIN)),
            failure_hours=("failure_hour", "sum"),

            I_batt_max_A=("I_batt_max_A", "first"),


            I_req_max_A=("I_BAT_A", lambda s: np.max(np.abs(s.to_numpy()))),
        )

        # Derived metrics
        summary["soc_min_fraction"] = summary["hours_soc_min"] / summary["hours_total"]
        summary["soc_full_fraction"] = summary["hours_soc_full"] / summary["hours_total"]
        summary["net_Ah"] = summary["surplus_Ah"] - summary["deficit_Ah"]

        return summary

    def compute_optimal_score(self, summary, w_batt=1.0, w_panel=1.0, w_auto=1.0):
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
            Weight for autonomy (smaller is better)

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with added 'score' column, sorted by score descending.
        """
        def compute_raw_score(row):
            """Compute unnormalized raw score."""
            # Normalize battery capacity (smaller = better -> invert)
            batt_norm = (row["C_batt_Ah"] - summary["C_batt_Ah"].min()) / \
                        (summary["C_batt_Ah"].max() - summary["C_batt_Ah"].min())
            batt_score = 1.0 - batt_norm

            # Normalize panel area (smaller = better -> invert)
            panel_norm = (row["panel_area_m2"] - summary["panel_area_m2"].min()) / \
                         (summary["panel_area_m2"].max() - summary["panel_area_m2"].min())
            panel_score = 1.0 - panel_norm

            # Normalize autonomy (smaller = better -> invert)
            auto_norm = safe_normalized_autonomy(
                row["autonomy_hours"],
                summary["autonomy_hours"].min(),
                summary["autonomy_hours"].max()
            )
            auto_score = 1.0 - auto_norm


            # Weighted raw score
            raw = (
                w_batt * batt_score +
                w_panel * panel_score +
                w_auto * auto_score
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

    def run_full_simulation(self, irradiance_filepath):
        """
        Run the complete simulation pipeline.

        Parameters
        ----------
        irradiance_filepath : str
            Path to the irradiance data CSV file

        Returns
        -------
        dict
            Dictionary containing:
            - 'df_pv': PV power data
            - 'df_pv_pmu': Power balance data
            - 'df_soc': SoC simulation data
            - 'summary': Summary metrics with scores
        """
        print("Loading irradiance data...")
        irr_data = self.load_irradiance_data(irradiance_filepath)

        print("Computing PV power...")
        df_pv = self.compute_pv_power(irr_data)

        print("Computing hourly balance...")
        df_pv_pmu = self.compute_hourly_balance(df_pv)

        print("Simulating battery SoC...")
        df_soc = self.simulate_battery_soc(df_pv_pmu)

        print("Evaluating viability...")
        summary = self.evaluate_viability(df_soc)

        print("Computing optimal scores...")
        summary = self.compute_optimal_score(summary)

        print("Done!")

        return {
            'df_pv': df_pv,
            'df_pv_pmu': df_pv_pmu,
            'df_soc': df_soc,
            'summary': summary
        }
