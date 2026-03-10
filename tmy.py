#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def sanitize_file_stem(city_name: str) -> str:
    cleaned = city_name.strip().replace(",", " ")
    parts = [chunk for chunk in cleaned.split() if chunk]
    return "_".join(parts)


def get_coordinates_for_city(city_name: str) -> tuple[float, float]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": city_name,
        "format": "json",
        "limit": 1,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    }

    response = requests.get(url, params=params, timeout=15, headers=headers)
    response.raise_for_status()
    data = response.json()

    if not data:
        raise ValueError(f"No coordinates were found for: {city_name}")

    return float(data[0]["lat"]), float(data[0]["lon"])


def get_orig_tmy_data(city_name: str, start_year: int, end_year: int) -> dict:
    lat, lon = get_coordinates_for_city(city_name)
    logger.info(f"Coordinates for {city_name}: lat={lat}, lon={lon}")

    url = "https://re.jrc.ec.europa.eu/api/tmy"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": start_year,
        "endyear": end_year,
        "outputformat": "json",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def build_adjusted_dataframe(hourly: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(hourly)
    if "time(UTC)" not in df.columns or "G(h)" not in df.columns or "T2m" not in df.columns:
        raise ValueError("API response does not contain expected columns: time(UTC), G(h), T2m")

    datetime_series = pd.to_datetime(df["time(UTC)"], format="%Y%m%d:%H%M", errors="raise")

    adjusted = pd.DataFrame({
        "Date-hour": df["time(UTC)"],
        "Month": datetime_series.dt.month.astype(str).str.zfill(2),
        "Day": datetime_series.dt.day.astype(str).str.zfill(2),
        "Hour": datetime_series.dt.hour.astype(str).str.zfill(2),
        "G(h)": df["G(h)"],
        "Temperature": df["T2m"],
    })

    return adjusted


def get_tmy_csv(
    city_name: str,
    start_year: int,
    end_year: int,
    output_dir: str = "raw-data",
) -> str:

    if start_year > end_year:
        raise ValueError("start_year cannot be greater than end_year")

    file_stem = f"{sanitize_file_stem(city_name)}_{start_year}_{end_year}"
    output_dir_path = Path(output_dir)
    adjusted_path = output_dir_path / f"{file_stem}.csv"

    if adjusted_path.exists():
        logger.info(f"CSV already exists, skipping download: {adjusted_path}")
        return str(adjusted_path)

    data = get_orig_tmy_data(city_name, start_year, end_year)
    logger.info(json.dumps(data['inputs'], indent=2))

    hourly = data["outputs"]["tmy_hourly"]
    adjusted_df = build_adjusted_dataframe(hourly)

    output_dir_path.mkdir(parents=True, exist_ok=True)
    adjusted_df.to_csv(adjusted_path, index=False)

    return str(adjusted_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PVGIS TMY data for a city and generate a simulation-ready CSV file."
    )
    parser.add_argument("city", help="City to query. Example: 'Quito, Ecuador'")
    parser.add_argument("--start-year", type=int, default=2007, help="Start year for PVGIS (default: 2007)")
    parser.add_argument("--end-year", type=int, default=2020, help="End year for PVGIS (default: 2020)")
    parser.add_argument(
        "--output-dir",
        default="raw-data",
        help="Output directory for CSV files (default: raw-data)",
    )
    args = parser.parse_args()

    adjusted_path = get_tmy_csv(
        city_name=args.city,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
    )

    logger.info(f"Adjusted CSV saved to: {adjusted_path}")


if __name__ == "__main__":
    main()
