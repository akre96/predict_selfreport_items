"""
Quality control for sensor features

inputs:
    - sensor features: output from collect_sensor_features.py
outputs:
    - QC'd sensor features csv
"""

import argparse
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensor-features",
        type=str,
        help="csv file containing sensor features",
    )
    parser.add_argument(
        "--output-file", type=str, help="csv file to write processed data to"
    )
    args = parser.parse_args()

    # Check that all files exist
    for item in [args.sensor_features]:
        assert Path(item).exists(), f"{item} does not exist"

    sensor_features = pd.read_csv(args.sensor_features)
    qc_features = sensor_features[
        (sensor_features.QC_watch_on_percent > 80)
        & (sensor_features.QC_duration_days < 40)
        & (sensor_features.QC_ndates > 4)
    ].copy()
    qc_features = qc_features.dropna(subset=["sleep_sleepDuration_day_median"])
    qc_features.loc[
        qc_features.sleep_Awake_mean.isna(), "sleep_Awake_mean"
    ] = 0
    print("Sensor features shape:", sensor_features.shape)
    print("QC features shape:", qc_features.shape)

    print("Saving QC features to", args.output_file)
    qc_features.to_csv(args.output_file, index=False)
