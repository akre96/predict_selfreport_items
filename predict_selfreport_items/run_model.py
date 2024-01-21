import pickle
from pathlib import Path
import pandas as pd
import argparse

FEATURES = [
    "watch_on_hours",
    "sleep_sleepDuration_day_median",
    "sleep_sleepEfficiency_day_median",
    "sleep_sleepEfficiency_day_std",
    "sleep_Awake_mean",
    "sleep_sleepOnsetLatency_day_median",
    "sleep_sleepOffsetHours_day_std",
    "sleep_bedrestOffsetHours_day_std",
    "sleep_bedrestDuration_day_median",
    "sleep_sleepHR_day_median",
    "sleep_sleepHRV_day_median",
    "ActiveEnergyBurned_sum",
    "ActiveEnergyBurned_duration",
    "StepCount_sum",
    "HeartRate_median",
    "HeartRate_circadian_period",
    "HeartRateVariabilitySDNN_median",
    "HeartRateVariabilitySDNN_circadian_period",
    "RespiratoryRate_median",
    "RespiratoryRate_circadian_period",
]


def predict_selfreport_items(
    sensor_features_path: str | Path,
    model_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    Predict self-report items using sensor features

    inputs:
        - sensor_features: output from collect_sensor_features.py
        - model_file: trained model file
        - output_file: csv file to write predictions to
    outputs:
        - csv file with predicted self-report items to path of `output_file`
    """
    # Check that all files exist
    for item in [sensor_features_path, model_file]:
        assert Path(item).exists(), f"{item} does not exist"

    # Load model
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    feature_cols = model.feature_names_in_

    # Load sensor features
    sensor_features = pd.read_csv(sensor_features_path)
    print("Sensor features shape:", sensor_features.shape)
    sensor_features = sensor_features.dropna(subset=feature_cols)

    # Predict self-report items
    print("Predicting self-report items...")
    predictions = model.predict(sensor_features[feature_cols])
    print("Predictions shape:", predictions.shape)

    # Save predictions
    print("Saving predictions to", output_file)
    sensor_features["prediction"] = predictions
    sensor_features.to_csv(output_file, index=False, header=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensor-features",
        type=str,
        help="csv file containing sensor features",
    )
    parser.add_argument("--model-file", type=str, help="trained model file")
    parser.add_argument(
        "--output-file", type=str, help="csv file to write predictions to"
    )
    args = parser.parse_args()

    predict_selfreport_items(
        args.sensor_features, args.model_file, args.output_file
    )
