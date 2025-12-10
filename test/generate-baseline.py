#!/usr/bin/env python3
import json
import hashlib
import pandas as pd
from simulator import Simulator, Config

def df_hash(df):
    m = hashlib.md5()
    m.update(
        df.to_csv(index=False).encode("utf-8")
    )
    return m.hexdigest()

def main():
    simulator = Simulator()
    result = simulator.run_full_simulation("raw-data/CR.csv")

    df_soc = result["df_soc"]
    summary = result["summary"]

    baseline = {
        "soc_stats": {
            "mean": float(df_soc["SoC"].mean()),
            "std": float(df_soc["SoC"].std()),
            "min": float(df_soc["SoC"].min()),
            "max": float(df_soc["SoC"].max()),
        },
        "ibat_stats": {
            "mean": float(df_soc["I_BAT_A"].mean()),
            "std": float(df_soc["I_BAT_A"].std()),
            "min": float(df_soc["I_BAT_A"].min()),
            "max": float(df_soc["I_BAT_A"].max()),
        },
        "summary_top1": summary.iloc[0].to_dict(),
        "summary_hash": df_hash(
            summary[[
                "panel_area_m2",
                "C_batt_Ah",
                "eta_PMU",
                "autonomy_hours",
                "failure_hours",
                "score"
            ]]
        ),
    }

    with open("test/baseline_simulation.json", "w") as f:
        json.dump(baseline, f, indent=2)

if __name__ == "__main__":
    main()
