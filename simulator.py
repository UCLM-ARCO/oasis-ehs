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

    PMU_ETA_VALUES = [0.70, 0.75, 0.80, 0.85, 0.87, 0.90, 0.95, 0.98]

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
        0.010057, # 101 cm² (=11.3x8.9cm) https://github.com/VoltaicEngineering/Solar-Panel-Drawings/blob/master/Voltaic%20Systems%201W%206V%20113x89mm%20DRAWING%20CURRENT%202017%207%2020.pdf
        0.0160,   # 160 cm² (≈12.6×12.6 cm)
        0.0250,   # 250 cm² (≈15.8×15.8 cm)
        0.030414, # 304 cm²,(13.7 x 22.2) https://www.adafruit.com/product/5367
        0.0310,   # 310 cm² (≈17.6×17.6 cm)
        0.0385,   # 385 cm² (=22x17.5cm) https://www.adafruit.com/product/1525
        0.0400,   # 400 cm² (≈20×20 cm)
        0.056797, # 568 cm² (=22.1x25.7 cm) https://github.com/VoltaicEngineering/Solar-Panel-Drawings/blob/master/Voltaic%20Systems%209W%206V%20221x257mm%20DRAWING%20CURRENT%202017%207%2020.pdf
        0.061102  # 611 cm² (=22.3x27.4 cm) https://www.adafruit.com/product/5369
    ]

    # Melisa
    # https://docs.google.com/spreadsheets/d/16a1q3lofZJNS3nguHSokKgYhDOOP0ylY17oNx9dazzg/edit?gid=0#gid=0
    BATTERY_SPECS = [
        BatterySpec(0.011, 0.011, "LiPo",
                    "11 mAh  — SHENZHEN PKCELL — https://www.eemb.com/product-32"),

        BatterySpec(0.105, 0.1575, "LiPo",
                    "105 mAh (401230) — SHENZHEN PKCELL — https://cdn-shop.adafruit.com/product-files/2750/LP552035_350MAH_3.7V_20150906.pdf"),

        BatterySpec(0.120, 0.2400, "Li-ion",
                    "120 mAh coin cell (LIR2450) — https://cdn-shop.adafruit.com/datasheets/LIR2450.pdf"),

        BatterySpec(0.250, 0.250, "Li-ion",
                    "250mAh https://www.amazon.es/dp/B08FD3V6TF?ref=emc_s_m_5_i_atc&th=1"),

        BatterySpec(0.350, 0.5250, "LiPo",
                    "350 mAh — https://cdn-shop.adafruit.com/product-files/2750/LP552035_350MAH_3.7V_20150906.pdf"),

        BatterySpec(0.360, 0.720, "LiPo",
                    "350 mAh — https://www.ebay.it/itm/314935725680"),

        BatterySpec(0.400, 0.6000, "LiPo",
                    "400 mAh — https://cdn-shop.adafruit.com/product-files/3898/3898_specsheet_LP801735_400mAh_3.7V_20161129.pdf"),

        BatterySpec(0.420, 0.6300, "LiPo",
                    "420 mAh — https://cdn-shop.adafruit.com/product-files/4236/4236_ds_LP552535+420mAh+3.7V.pdf"),

        BatterySpec(0.500, 0.5000, "LiPo",
                    "500 mAh — https://cdn-shop.adafruit.com/product-files/1578/Datasheet.pdf"),

        BatterySpec(0.550, 1.000, "LiPo",
                    "550 mAh — https://eemb.store/products/lp602835-3-7v-550mah"),

        BatterySpec(0.700, 0.700, "LiPo",
                    "700 mAh - https://support.pluxbiosignals.com/wp-content/uploads/2021/11/lp553436-datasheet.pdf "),

        BatterySpec(0.750, 1.500, "LiPo",
                    "750 mAh - https://macrogroup.ru/upload/iblock/215/j606vmmes1heftrm2qjhf7v50oz78yd6/LP523048.pdf"),

        BatterySpec(0.880, 1.760, "LiPo",
                    "880 mAh -  https://macrogroup.ru/upload/iblock/bcf/jppp6b0xsz9q5cxigudx913qc3rzfudg/LP503448_880mAh.pdf"),

        BatterySpec(1.100, 2.0000, "LiPo",
                    "1100 mAh — https://www.ebay.co.uk/itm/257173548788?customid&toolid=10050"),

        BatterySpec(1.200, 1.2000, "LiPo",
                    "1200 mAh — https://cdn-shop.adafruit.com/product-files/258/C101-_Li-Polymer_503562_1200mAh_3.7V_with_PCM_APPROVED_8.18.pdf%22"),

        BatterySpec(1.400, 2.000, "LiPo",
                    "1400 mAh — https://www.amazon.es/dp/B095W4HS75?ref=emc_s_m_5_i_atc&th=1"),

        BatterySpec(1.600, 4.8000, "LiFePO4",
                    "1600 mAh (high-discharge) — https://www.antbatt.com/wp-content/uploads/2019/09/18650-3.2V-1600mAh-datasheet.pdf"),

        BatterySpec(1.800, 3.6000, "LiPo",
                    "1800 mAh — https://macrogroup.ru/upload/iblock/34a/ar3b90i9orrr8q4pawtb1tphma0gtgt1/LP103448.pdf"),

        BatterySpec(2.000, 2.0000, "LiPo",
                    "2000 mAh — https://cdn-shop.adafruit.com/datasheets/LiIon2000mAh37V.pdf"),

        BatterySpec(2.300, 46.0000, "LiFePO4",
                    "2300 mAh (very high-discharge) — https://docs.rs-online.com/4ad1/0900766b812fdd10.pdf"),

        BatterySpec(2.500, 1.5000, "LiPo",
                    "2500 mAh — https://cdn-shop.adafruit.com/product-files/328/LP785060+2500mAh+3.7V+20190510.pdf"),

        BatterySpec(3.200, 9.6000, "LiFePO4",
                    "3200 mAh (high-discharge) — https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/180/P3_2D00_Datasheet-Cell--3232-LFP-26650.pdf%22")
    ]

    score_objectives = [
        ("C_batt_Ah",         -1),
        ("panel_area_m2",     -1),
        ("soc_full_fraction", -1),
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
        self.config = config or Config()

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

        return pd.DataFrame(design_space, columns=[
            "panel_area_m2", "battery_capacity_Ah", "eta_PMU"
        ])

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
        df = pd.concat(
            [df_pv.assign(eta_PMU=eta) for eta in self.config.PMU_ETA_VALUES],
            ignore_index=True
        )

        # PV current before PMU -> I_PV_raw_A = P_PV / Vbat
        df["I_PV_raw_A"] = df["P_PV"] / self.config.BATTERY_VOLTAGE

        # PV current available to the load (η_PMU applies)
        df["I_PV_to_load_A"] = df["I_PV_raw_A"] * df["eta_PMU"]

        # Load current (constant power node)
        I_load_A = self.config.NODE_POWER_W / self.config.BATTERY_VOLTAGE
        df["I_load_A"] = I_load_A

        # Net current needed from battery:
        #   net > 0 => panel insufficient => battery DISCHARGES
        #   net < 0 => panel exceeds load => surplus charges battery
        net = df["I_load_A"] - df["I_PV_to_load_A"]

        # Battery current with PMU efficiency:
        #   Discharge:  I_BAT_A =  net / η_PMU     (> 0)
        #   Charge:     I_BAT_A =  net * η_PMU     (< 0)
        df["I_BAT_A"] = np.where(
            net >= 0.0,
            net / df["eta_PMU"],
            net * df["eta_PMU"]
        )

        return df

    def simulate_battery_soc(self, df):
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
                df.assign(C_batt_Ah=spec.capacity_Ah,
                          I_batt_max_A=spec.max_discharge_A)
                for spec in self.config.BATTERY_SPECS
            ],
            ignore_index=True
        )

        df_soc["SoC"] = np.nan

        grouped = df_soc.groupby(["panel_area_m2", "C_batt_Ah", "eta_PMU"])

        for (_, Cbat, _), idx in grouped.groups.items():
            g = df_soc.loc[idx].sort_values("hour_index")
            soc = simulate_soc_kernel(
                g["I_BAT_A"].to_numpy().astype(np.float64),
                float(Cbat),
                self.config.SOC_MIN,
                self.config.BATTERY_ETA_C
            )
            df_soc.loc[g.index, "SoC"] = soc

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

    def compute_score(self, summary, objectives):
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
        objectives : list of (str, int | float)
            Each element is a ``(column_name, criterion)`` pair where:

            - ``criterion > 0`` → higher is better (maximise).
            - ``criterion < 0`` → lower is better (minimise).
            - ``abs(criterion)`` is the relative weight.

            Example: ``("C_batt_Ah", -2)`` means minimise battery capacity
            with double weight compared to an objective using ``-1``.

            This parameter is mandatory and must be non-empty.

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with added 'score' column, sorted by score descending.

        """

        if not objectives:
            raise ValueError(
                "'objectives' is mandatory and must contain at least one "
                "(column_name, criterion) pair."
            )

        def norm_col(series, value, direction):
            lo, hi = series.min(), series.max()
            den = hi - lo
            if den <= 0:
                return 1.0
            normalized = (value - lo) / den
            return normalized if direction > 0 else 1.0 - normalized

        def compute_raw_score(row):
            total = 0.0
            for col, criterion in objectives:
                direction = 1 if criterion >= 0 else -1
                weight = abs(float(criterion))
                total += weight * norm_col(summary[col], row[col], direction)
            return total

        # Compute raw scores for all rows
        summary["raw_score"] = summary.apply(compute_raw_score, axis=1)

        # Normalize raw_score to [0,1]
        raw_min = summary["raw_score"].min()
        raw_max = summary["raw_score"].max()

        summary["score"] = (summary["raw_score"] - raw_min) / (raw_max - raw_min + 1e-12)

        # Apply strict penalty for failures
        # Any configuration with failure_hours > 0 gets score = 0
        summary.loc[summary["failure_hours"] > 0, "score"] = 0.0

        summary = summary.drop(columns="raw_score")

        # Sort table by score (best first)
        summary = summary.sort_values("score", ascending=False).reset_index(drop=True)

        return summary

    def pareto_front(self, summary, objectives):
        """
        Return the Pareto-dominant configurations from the summary DataFrame.

        A configuration A dominates B if A is at least as good as B in all
        objectives and strictly better in at least one.

        Only viable configurations (failure_hours == 0) are considered.

        Parameters
        ----------
        summary : pd.DataFrame
            Output of ``evaluate_viability()`` or ``compute_optimal_score()``.
        objectives : list of (str, int | float)
            Each element is a ``(column_name, criterion)`` pair where the
            **sign** of *criterion* sets the optimisation direction:
            ``criterion > 0`` → maximise, ``criterion < 0`` → minimise.
            The magnitude is ignored for dominance; only the sign matters.

        Returns
        -------
        pd.DataFrame
            Subset of *summary* containing only Pareto-dominant rows,
            reset-indexed and sorted by the first objective column.

        Examples
        --------
        Minimise battery and panel, maximise autonomy::

            sim.pareto_front(
                summary,
                objectives=[
                    ("C_batt_Ah",      -1),
                    ("panel_area_m2",  -1),
                    ("autonomy_hours", +1),
                ],
            )
        """
        cols     = [col for col, _ in objectives]
        criteria = [c   for _,   c in objectives]

        # Only viable configurations
        viable = summary[summary["failure_hours"] == 0].copy()
        if viable.empty:
            return viable

        # Build objective matrix — all converted to minimisation by negating
        # maximisation objectives (criterion > 0 → negate to minimise)
        obj_matrix = np.column_stack([
            viable[col].to_numpy() * (-1 if d > 0 else 1)
            for col, d in zip(cols, criteria)
        ])

        n = len(obj_matrix)
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # j dominates i: j <= i in all objectives and < in at least one
                if np.all(obj_matrix[j] <= obj_matrix[i]) and np.any(obj_matrix[j] < obj_matrix[i]):
                    is_dominated[i] = True
                    break

        front = viable.iloc[~is_dominated].sort_values(cols[0]).reset_index(drop=True)
        return front

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
        irr = self.load_irradiance_data(irradiance_filepath)

        print("Computing PV power...")
        df_pv = self.compute_pv_power(irr)

        print("Computing hourly balance...")
        df_bal = self.compute_hourly_balance(df_pv)

        print("Simulating battery SoC...")
        df_soc = self.simulate_battery_soc(df_bal)

        print("Evaluating viability...")
        summary = self.evaluate_viability(df_soc)

        print("Computing optimal scores...")

        summary = self.compute_score(summary, self.config.score_objectives)
        print("Done!")

        return {
            "df_pv": df_pv,
            "df_pv_pmu": df_bal,
            "df_soc": df_soc,
            "summary": summary,
        }
