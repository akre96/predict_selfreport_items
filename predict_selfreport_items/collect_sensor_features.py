"""
Author: Samir Akre

This script aggregates digital health data prior to a timestamp for a given duration per participant.

Inputs:
    - survey_data: csv file containing survey data processed with process_survey_data.py
    - sensor_data_folder: folder containing sensor data, one csv file per participant, labeled with <ethica ID>.csv
    - output_file: csv file to write processed data to
    - surveys: list of surveys to include in the output
    - parallel: whether to run in parallel using all available CPUs

Example:
python predict_selfreport_items/collect_sensor_features.py \
    --survey_data test.csv \
    --sensor_data_folder ~/Data/OPTIMA/cassandra_HK_participant \
    --output_file sensordata.csv \
    --surveys psqi phq14 pvss \
    --parallel
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from mhealth_feature_generation import simple_features, dataloader
from tqdm import tqdm
from p_tqdm import p_map
import warnings


def get_hkmetrics(user_hk, user_id, ts, duration):
    hk_window_data = simple_features.getDurationAroundTimestamp(
        user_hk, user_id, ts, duration=duration
    )
    if hk_window_data.empty:
        return pd.DataFrame()

    # Aggregate features
    agg_sleep_features = [
        "sleep_sleepDuration_day",
        "sleep_bedrestDuration_day",
        "sleep_sleepHR_day",
        "sleep_sleepHRV_day",
        "sleep_wakeAfterSleepOnset_day",
        "sleep_sleepEfficiency_day",
        "sleep_sleepOnsetLatency_day",
        "sleep_bedrestOnsetHours_day",
        "sleep_bedrestOffsetHours_day",
        "sleep_sleepOnsetHours_day",
        "sleep_sleepOffsetHours_day",
        "sleep_sleepNoise_day",
    ]
    sleep_agg = simple_features.aggregateDailySleep(
        hk_window_data,
        aggs=["min", "median", "max", "std"],
        sleep_features=agg_sleep_features,
    )
    sleep_cat_agg = simple_features.aggregateSleepCategories(
        hk_window_data,
    )
    paee = simple_features.aggregateActiveDuration(
        hk_window_data, "ActiveEnergyBurned"
    )
    etime = simple_features.aggregateActiveDuration(
        hk_window_data, "AppleExerciseTime"
    )
    steps = simple_features.aggregateActiveDuration(
        hk_window_data, "StepCount"
    )
    hr = simple_features.aggregateVital(
        hk_window_data,
        "HeartRate",
        resample="1h",
        standard_aggregations=["min", "median", "max", "std"],
        linear_time_aggregations=True,
        circadian_model_aggregations=True,
        vital_range=(30, 200),
    )
    noise = simple_features.aggregateAudioExposure(
        hk_window_data,
        resample="1h",
    )
    hrv = simple_features.aggregateVital(
        hk_window_data,
        "HeartRateVariabilitySDNN",
        resample="1h",
        standard_aggregations=["min", "median", "max", "std"],
        linear_time_aggregations=True,
        circadian_model_aggregations=True,
        vital_range=(0, 1),
    )
    o2sat = simple_features.aggregateVital(
        hk_window_data,
        "OxygenSaturation",
        resample="1h",
        standard_aggregations=["min", "median", "max", "std"],
        linear_time_aggregations=True,
        vital_range=(0.5, 1),
    )
    rr = simple_features.aggregateVital(
        hk_window_data,
        "RespiratoryRate",
        resample="1h",
        standard_aggregations=["min", "median", "max", "std"],
        linear_time_aggregations=True,
        circadian_model_aggregations=True,
        vital_range=(0.1, 100),
    )
    hk_metrics = pd.concat(
        [
            sleep_agg,
            sleep_cat_agg,
            paee,
            etime,
            steps,
            hr,
            hrv,
            o2sat,
            rr,
            noise,
        ],
        axis=1,
    )

    # Add QC metrics
    start, end = simple_features.calcStartStop(ts, duration)
    if end != ts:
        print("Warning: end of window does not match end of data")

    hk_metrics["QC_watch_on_percent"] = simple_features.processWatchOnPercent(
        hk_window_data, resample="1h", origin=start, end=ts
    )
    hk_metrics["QC_watch_on_hours"] = simple_features.processWatchOnTime(
        hk_window_data, resample="1h", origin=start
    )
    hk_metrics["QC_duration_days"] = (
        hk_window_data["local_start"].max()
        - hk_window_data["local_start"].min()
    ) / np.timedelta64(1, "D")
    hk_metrics["QC_ndates"] = hk_window_data["local_start"].dt.date.nunique()

    # Add metadata
    hk_metrics["user_id"] = user_id
    hk_metrics["survey_start"] = ts
    hk_metrics["expected_duration"] = duration

    return hk_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--survey-data",
        type=str,
        help="csv file containing survey data processed with process_survey_data.py",
        default="~/Data/OPTIMA/survey_data_Aug032023.csv",
    )
    parser.add_argument(
        "--sensor-data-folder",
        type=str,
        help="folder containing sensor data, one folder per sensor, one file per participant-sensor labeled with <ethica ID>-<sensor>.csv",
        default="~/Data/OPTIMA/HealthKit_Datastreams",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="csv file to write processed data to",
        default="sensordata.csv",
    )
    parser.add_argument(
        "--surveys",
        nargs="+",
        help="list of surveys to include in the output",
        default=["phq14", "pvss", "psqi"],
    )
    parser.add_argument(
        "--parallel", action="store_true", help="whether to run in parallel"
    )
    parser.add_argument(
        "--debug", action="store_true", help="whether to run in debug mode"
    )

    args = parser.parse_args()
    # check that all files exist
    for item in [args.survey_data, args.sensor_data_folder]:
        assert Path(item).expanduser().exists(), f"{item} does not exist"
    sensor_data_folder = Path(args.sensor_data_folder)
    sensor_folders = [
        f for f in sensor_data_folder.expanduser().iterdir() if f.is_dir()
    ]

    # Filter for relevant surveys
    survey_data = pd.read_csv(args.survey_data, parse_dates=["survey_start"])

    # Tester accounts
    exclude_ids = [7, 10, 17, 23, 26, 50]

    # Exclude tester accounts
    ids = [
        int(f) for f in survey_data.user_id.unique() if f not in exclude_ids
    ]
    if args.debug:
        ids = [150, 130, 82, 51, 40, 63]
    relevant_surveys = survey_data[
        survey_data["survey"].isin(args.surveys)
        & (survey_data.redcap_event_name != "baseline_arm_1")
        & (survey_data.user_id.isin(ids))
    ]
    if relevant_surveys.empty:
        raise ValueError(
            f"No surveys found in {args.survey_data} for {args.surveys}"
        )
    relevant_surveys["user_id"] = relevant_surveys["user_id"].astype(int)

    print("Making PHQ and PVSS all 1 week aggregations")
    relevant_surveys.loc[
        relevant_surveys.survey.isin(["phq14", "pvss"]), "duration"
    ] = "7days"

    # Get unique timestamps and durations for aggregations
    timepoint_df = (
        relevant_surveys[["user_id", "survey_start", "duration"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    hk_fs = []  # Hold healthkit features
    # Gather healthkit features from the relevant duration prior to a timestamp per participant
    loader = dataloader.DataLoader()
    for uid, u_df in tqdm(timepoint_df.groupby("user_id")):
        hk_data_list = []
        for sensor_folder in sensor_folders:
            sensor_path = Path(
                sensor_folder, f"{int(uid)}-{sensor_folder.name}.csv"
            )
            if not sensor_path.exists():
                continue
            hk_data_list.append(loader.loadData(sensor_path))
        if not hk_data_list:
            print(f"Skipping {uid} due to no healthkit data")
            continue
        hk_data = pd.concat(hk_data_list)
        if hk_data.empty:
            print(f"Skipping {uid} due to no healthkit data")
            continue

        def HKMetricWrapper(inputs):
            user, survey_start, duration = inputs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                feature_data = get_hkmetrics(
                    hk_data, user, survey_start, duration
                )
            feature_data["duration"] = duration

            return feature_data

        inputs = [
            (uid, survey_start, duration)
            for survey_start, duration in u_df[
                ["survey_start", "duration"]
            ].values
        ]
        if args.parallel:
            metrics = pd.concat(
                p_map(HKMetricWrapper, inputs, desc=f"User {uid}")
            )
        else:
            metrics = pd.concat(
                [HKMetricWrapper(i) for i in tqdm(inputs, desc=f"User {uid}")]
            )
        hk_fs.append(metrics)
    hk_features = pd.concat(hk_fs)

    print("Writing healthkit features to", args.output_file)
    hk_features.to_csv(args.output_file, index=False)
