"""
Visualization functions for OASIS panel simulation analysis.

This module provides plotting functions for analyzing solar panel and battery
configurations, including State of Charge (SoC) analysis, irradiance patterns,
battery current, and seasonal variations.
"""

import matplotlib.pyplot as plt


def get_config_trace(df_soc, panel, capacity, eta):
    """
    Filter df_soc for one configuration and return it sorted by hour_index.

    Parameters
    ----------
    df_soc : DataFrame
        Simulation results with SoC data.
    panel : float
        Panel area (m²).
    capacity : float
        Battery capacity (Ah).
    eta : float
        PMU efficiency.

    Returns
    -------
    DataFrame
        Filtered and sorted configuration data.
    """
    cfg = df_soc[
        (df_soc["panel_area_m2"] == panel) &
        (df_soc["C_batt_Ah"] == capacity) &
        (df_soc["eta_PMU"] == eta)
    ].sort_values("hour_index")
    return cfg


def plot_soc(df_soc, panel, capacity, eta, hours=240, title=None):
    """
    Plot SoC evolution for a specific configuration.
    Default: first 240 hours (~10 days)
    """
    cfg = df_soc[
        (df_soc["panel_area_m2"] == panel) &
        (df_soc["C_batt_Ah"] == capacity) &
        (df_soc["eta_PMU"] == eta)
    ].sort_values("hour_index")

    window = cfg[cfg["hour_index"] < hours]

    if title is None:
        title = f"A={panel} m², C={capacity} Ah, η={eta}"

    plt.figure(figsize=(14,4))
    plt.plot(window["hour_index"], window["SoC"])
    plt.ylim(0,1.05)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Hour index")
    plt.ylabel("State of Charge (SoC)")
    plt.title(title)
    plt.show()


def plot_soc_gh(cfg, df_pv, hours=240, title=None):
    """import matplotlib.pyplot as plt

    Plot SoC and irradiance (G_h) for a given configuration.

    Parameters
    ----------
    cfg : DataFrame
        Configuration data with SoC values.
    df_pv : DataFrame
        PV data with irradiance (G_h).
    hours : int, optional
        Number of hours to display (default 240).
    title : str, optional
        Plot title.
    """
    # Slice hours
    cfg_win = cfg[cfg["hour_index"] < hours]
    pv_win = df_pv[df_pv["panel_area_m2"] == cfg["panel_area_m2"].iloc[0]]
    pv_win = pv_win[pv_win["hour_index"] < hours]

    plt.figure(figsize=(14, 5))

    # SoC (primary axis)
    plt.plot(cfg_win["hour_index"], cfg_win["SoC"], label="SoC", linewidth=2)

    # Irradiance (secondary axis)
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(pv_win["hour_index"], pv_win["G_h"], color="orange", alpha=0.5, label="G_h (W/m²)")

    plt.title(title or "SoC + G_h")
    ax.set_xlabel("Hour index")
    ax.set_ylabel("SoC")
    ax2.set_ylabel("G_h (W/m²)")

    ax.grid(True, alpha=0.3)

    plt.show()


def plot_soc_ibatt(cfg, hours=240, title=None):
    """
    Plot SoC and battery current (I_BAT_A) for a given configuration.

    Parameters
    ----------
    cfg : DataFrame
        Configuration data with SoC and I_BAT_A values.
    hours : int, optional
        Number of hours to display (default 240).
    title : str, optional
        Plot title.
    """
    cfg_win = cfg[cfg["hour_index"] < hours]

    plt.figure(figsize=(14, 6))

    # SoC curve
    ax = plt.gca()
    ax.plot(cfg_win["hour_index"], cfg_win["SoC"], label="SoC", linewidth=2)

    # Battery current
    ax2 = ax.twinx()
    ax2.plot(cfg_win["hour_index"], cfg_win["I_BAT_A"],
             color="red", alpha=0.6, label="I_BAT_A")

    ax.set_xlabel("Hour index")
    ax.set_ylabel("SoC")
    ax2.set_ylabel("I_BAT_A (A)")

    plt.title(title or "SoC + battery current")
    ax.grid(True, alpha=0.3)

    plt.show()


def plot_daily(cfg, day_index=0, title=None):
    """
    Plot a single day (24 hours) of SoC.

    Parameters
    ----------
    cfg : DataFrame
        Configuration data with SoC values.
    day_index : int, optional
        Which day to plot (0 = hours 0-23, 1 = hours 24-47, etc.).
    title : str, optional
        Plot title.
    """
    start = day_index * 24
    end = start + 24

    win = cfg[(cfg["hour_index"] >= start) & (cfg["hour_index"] < end)]

    plt.figure(figsize=(12, 4))
    plt.plot(win["Hour"], win["SoC"], linewidth=2)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Hour of day")
    plt.ylabel("SoC")
    plt.title(title or f"SoC during day {day_index}")
    plt.show()


def plot_seasonal_soc(cfg):
    """
    Plot average SoC by season: Winter, Spring, Summer, Autumn.

    Parameters
    ----------
    cfg : DataFrame
        Configuration data with Month and SoC columns.
    """
    # Month → Season mapping
    season = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }

    df = cfg.copy()
    df["Season"] = df["Month"].map(season)

    season_avg = df.groupby("Season")["SoC"].mean().reindex(
        ["Winter", "Spring", "Summer", "Autumn"]
    )

    plt.figure(figsize=(8, 4))
    plt.bar(season_avg.index, season_avg.values)
    plt.ylim(0, 1)
    plt.title("Average SoC by season")
    plt.ylabel("Average SoC")
    plt.show()


def plot_full_soc_analysis(title, df_soc, df_pv, panel, capacity, eta, hours=240, day_index=0):
    """
    Generate a 4-panel figure showing comprehensive SoC analysis.

    Creates a figure with:
    1) SoC + Irradiance (G_h)
    2) SoC + Battery current (I_BAT_A)
    3) Daily SoC evolution (24 h)
    4) Seasonal average SoC

    Parameters
    ----------
    title : str
        Main title for the figure.
    df_soc : DataFrame
        Simulation results with SoC and I_BAT_A.
    df_pv : DataFrame
        PV data with irradiance (G_h).
    panel : float
        Panel area (m²).
    capacity : float
        Battery capacity (Ah).
    eta : float
        PMU efficiency.
    hours : int, optional
        Number of hours to show in time-series plots (default 240 = 10 days).
    day_index : int, optional
        Which day to plot (24h window).
    """
    # --- Select configuration ---
    cfg = df_soc[
        (df_soc["panel_area_m2"] == panel) &
        (df_soc["C_batt_Ah"] == capacity) &
        (df_soc["eta_PMU"] == eta)
    ].sort_values("hour_index")

    pv_cfg = df_pv[df_pv["panel_area_m2"] == panel]
    cfg_win = cfg[cfg["hour_index"] < hours]
    pv_win = pv_cfg[pv_cfg["hour_index"] < hours]

    # Day selection
    start = day_index * 24
    end = start + 24
    day = cfg[(cfg["hour_index"] >= start) & (cfg["hour_index"] < end)]

    # Season mapping
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    cfg["Season"] = cfg["Month"].map(season_map)
    season_avg = cfg.groupby("Season")["SoC"].mean().reindex(
        ["Winter", "Spring", "Summer", "Autumn"]
    )

    # ---------------------------------------------------------
    # Create 4-panel figure
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    (ax1, ax2), (ax3, ax4) = axs

    # ---------------------------------------------------------
    # 1) SoC + irradiance
    # ---------------------------------------------------------
    ax1.plot(cfg_win["hour_index"], cfg_win["SoC"], label="SoC", linewidth=2)
    ax1.set_ylabel("SoC")
    ax1.set_title("SoC + Irradiance (G_h)")
    ax1.grid(True, alpha=0.3)

    ax12 = ax1.twinx()
    ax12.plot(pv_win["hour_index"], pv_win["G_h"], color="orange", alpha=0.5, label="G_h")
    ax12.set_ylabel("G_h (W/m²)")

    # ---------------------------------------------------------
    # 2) SoC + battery current
    # ---------------------------------------------------------
    ax2.plot(cfg_win["hour_index"], cfg_win["SoC"], label="SoC", linewidth=2)
    ax2.set_ylabel("SoC")
    ax2.set_title("SoC + Battery Current (I_BAT_A)")
    ax2.grid(True, alpha=0.3)

    ax22 = ax2.twinx()
    ax22.plot(cfg_win["hour_index"], cfg_win["I_BAT_A"], color="red", alpha=0.5, label="I_BAT_A")
    ax22.set_ylabel("I_BAT_A (A)")

    # ---------------------------------------------------------
    # 3) Daily SoC profile
    # ---------------------------------------------------------
    ax3.plot(day["Hour"], day["SoC"], linewidth=2)
    ax3.set_title(f"SoC during Day {day_index}")
    ax3.set_xlabel("Hour of day")
    ax3.set_ylabel("SoC")
    ax3.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # 4) Seasonal average SoC
    # ---------------------------------------------------------
    ax4.bar(season_avg.index, season_avg.values)
    ax4.set_ylim(0, 1)
    ax4.set_title("Average SoC by Season")
    ax4.set_ylabel("Average SoC")

    # Global title
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()
