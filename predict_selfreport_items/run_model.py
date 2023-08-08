import pickle
from pathlib import Path
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import argparse


def predict_selfreport_items(sensor_features, model_file, output_file):
    """
    Predict self-report items using sensor features

    inputs:
        - sensor features: output from collect_sensor_features.py
        - model file: trained model file
    outputs:
        - csv file with predicted self-report items
    """
    # Check that all files exist
    for item in [sensor_features, model_file]:
        assert Path(item).exists(), f"{item} does not exist"

    # Load model
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    feature_cols = model.feature_names_in_

    # Load sensor features
    sensor_features = pd.read_csv(sensor_features)
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
