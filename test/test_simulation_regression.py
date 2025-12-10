#!/usr/bin/env python3

import json
import hashlib
from simulator import Simulator

def df_hash(df):
    m = hashlib.md5()
    m.update(
        df.to_csv(index=False).encode("utf-8")
    )
    return m.hexdigest()

def load_baseline():
    with open("test/baseline_simulation.json") as f:
        return json.load(f)

def test_simulation_regression():
    baseline = load_baseline()
    simulator = Simulator()
    result = simulator.run_full_simulation("raw-data/CR.csv")

    df_soc = result["df_soc"]
    summary = result["summary"]

    # --- 1) Test SoC global statistics ---
    assert abs(df_soc["SoC"].mean() - baseline["soc_stats"]["mean"]) < 1e-12
    assert abs(df_soc["SoC"].std()  - baseline["soc_stats"]["std"])  < 1e-12
    assert abs(df_soc["SoC"].min()  - baseline["soc_stats"]["min"])  < 1e-12
    assert abs(df_soc["SoC"].max()  - baseline["soc_stats"]["max"])  < 1e-12

    # --- 2) Test battery current statistics ---
    assert abs(df_soc["I_BAT_mA"].mean() - baseline["ibat_stats"]["mean"]) < 1e-12
    assert abs(df_soc["I_BAT_mA"].std()  - baseline["ibat_stats"]["std"])  < 1e-12
    assert abs(df_soc["I_BAT_mA"].min()  - baseline["ibat_stats"]["min"])  < 1e-12
    assert abs(df_soc["I_BAT_mA"].max()  - baseline["ibat_stats"]["max"])  < 1e-12

    # --- 3) Test top-1 result identical ---
    top1 = summary.iloc[0].to_dict()
    for k, v in baseline["summary_top1"].items():
        assert top1[k] == v

    # --- 4) Test summary hash identical ---
    current_hash = df_hash(summary[[
        "panel_area_m2",
        "C_batt_mAh",
        "eta_PMU",
        "autonomy_hours",
        "failure_hours",
        "score"
    ]])
    assert current_hash == baseline["summary_hash"]
