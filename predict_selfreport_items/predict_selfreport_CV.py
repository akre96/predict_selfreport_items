"""
Cross validation to evaluate the performance of the self-report item prediction models

inputs:
    - processed survey data: output from process_survey_data.py
    - processed sensor features: output from sensor_feature_qc.py
    - binary_thresholds: csv file containing thresholds for binarizing survey responses
    - output folder: folder to write results to
        - Gini feature importance
        - performance metrics
        - statistical test to see if ROC AUC performance > 0.5
        - model_weights
"""

from sklearn.model_selection import (
    StratifiedGroupKFold,
    KFold,
    RandomizedSearchCV,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    f_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import shap
from shap.utils._exceptions import InvalidModelError
import argparse
import pandas as pd
import numpy as np
import pingouin as pg
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import Tuple
from joblib import Parallel, delayed
import multiprocessing

NUM_CORES = multiprocessing.cpu_count()
N_JOBS_CV = 5
N_PROC = 5
if N_PROC > NUM_CORES:
    N_PROC = NUM_CORES


def train_test_model(
    outcome_df: pd.DataFrame,
    hk_feature_df: pd.DataFrame,
    hk_features: list[str],
    demographics: pd.DataFrame | None = None,
    demog_features: list[str] = [],
    dropna_subset: list[str] = ["response_binary"],
    inner_cv=KFold(n_splits=5, shuffle=True, random_state=42),
    model_name: str = "lr",
    model: object = LogisticRegression(),
    param_grid: dict = {
        "feature_select__k": [10, "all"],
        "feature_select__score_func": [mutual_info_classif, f_classif],
    },
    n_folds: int = 10,
) -> Tuple[pd.DataFrame, object]:
    ml_data = hk_feature_df.merge(
        outcome_df, on=["user_id", "survey_start"], how="left", validate="1:1"
    ).dropna(subset=dropna_subset)

    # if feature_select__k min is greater than number of features, set to all
    if "feature_select__k" in param_grid:
        int_k = [
            k for k in param_grid["feature_select__k"] if isinstance(k, int)
        ]
        if "all" in param_grid["feature_select__k"]:
            if len(hk_features) < min(int_k):
                param_grid["feature_select__k"] = ["all"]

    input_features = hk_features
    if (demographics is not None) and (len(demog_features) > 0):
        ml_data = ml_data.merge(
            demographics[["user_id", *demog_features]],
            on="user_id",
            how="left",
        ).dropna(subset=demog_features)
        input_features = hk_features + demog_features

    X = ml_data[input_features].copy()
    y = ml_data["response_binary"].astype(int)
    pipe = Pipeline(
        [
            ("variance_threshold", VarianceThreshold()),
            ("imputer", SimpleImputer(strategy="median")),
            ("feature_select", SelectKBest(k="all")),
            ("scaler", RobustScaler(quantile_range=(5, 95))),
            ("model", model),
        ]
    )
    if len(hk_features) == 1:
        param_grid = {}
    if model_name == "lr":
        n_iter = 4
    elif len(param_grid) > 0:
        n_iter = 5
    else:
        n_iter = 1
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        scoring="average_precision",
        n_iter=n_iter,
        cv=inner_cv,
        refit="average_precision",
        random_state=0,
        n_jobs=N_JOBS_CV,
    )
    cv = StratifiedGroupKFold(n_splits=n_folds)

    # Get individual fold prediction errors, split by weeks since baseline
    preds = []
    for i, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=ml_data.user_id)
    ):
        random_search.fit(X.iloc[train_idx], y.iloc[train_idx])
        m = random_search.best_estimator_.named_steps["model"]

        if isinstance(m, DummyClassifier):
            shap_vals = np.zeros((len(test_idx), len(input_features)))
            selected_feature_names = input_features
        else:
            preprocess = Pipeline(random_search.best_estimator_.steps[:-1])
            # Get the feature selection transformer
            feature_select = random_search.best_estimator_.named_steps[
                "feature_select"
            ]

            # Get the boolean mask of selected features
            selected_mask = feature_select.get_support()

            # Get the feature names from the imputer
            thresholder = random_search.best_estimator_.named_steps[
                "variance_threshold"
            ]
            feature_names_after_thresholding = (
                thresholder.get_feature_names_out()
            )

            # Index into the feature names with the mask
            selected_feature_names = feature_names_after_thresholding[
                selected_mask
            ]
            if isinstance(m, RandomForestClassifier) or isinstance(
                m, GradientBoostingClassifier
            ):
                explainer = shap.TreeExplainer(m)
            elif isinstance(m, LogisticRegression):
                explainer = shap.LinearExplainer(
                    m, preprocess.transform(X.iloc[train_idx])
                )
            else:
                raise InvalidModelError(f"Model {m} not supported")
            raw_shap_vals = explainer.shap_values(
                preprocess.transform(X.iloc[test_idx])
            )
            if len(np.shape(raw_shap_vals)) == 3:
                raw_shap_vals = raw_shap_vals[:, :, 1]
            if isinstance(raw_shap_vals, list):
                raw_shap_vals = raw_shap_vals[1]
            temp_shap_df = pd.DataFrame(
                raw_shap_vals, columns=selected_feature_names
            )
            shap_df = temp_shap_df.reindex(
                columns=input_features, fill_value=0
            )
            shap_vals = shap_df.to_numpy()

        preds.append(
            pd.DataFrame(
                {
                    "y_true": y.iloc[test_idx],
                    "y_pred": random_search.predict(X.iloc[test_idx]),
                    "y_pred_proba": random_search.predict_proba(
                        X.iloc[test_idx]
                    )[:, 1],
                    "SHAP_values": list(shap_vals),
                    "redcap_event_name": ml_data.iloc[
                        test_idx
                    ].redcap_event_name,
                    "survey_start": ml_data.iloc[test_idx].survey_start,
                    "user_id": ml_data.iloc[test_idx].user_id,
                    "fold": [i] * len(test_idx),
                    **{f: ml_data.iloc[test_idx][f] for f in input_features},
                }
            )
        )

    predictions = pd.concat(preds)
    predictions["model"] = model_name
    full_model = random_search.fit(X, y)
    return predictions, full_model.best_estimator_


def train_test_item_wrapper(
    item_df: pd.DataFrame,
    sensor_features: pd.DataFrame,
    output_folder: Path | str,
    feature_cols: list[str],
    model_paramgrid: list[tuple[str, object, dict]],
    nona_cols: list[str] = ["response_binary"],
    n_folds: int = 10,
    rerun_cached: bool = False,
) -> pd.DataFrame:
    if item_df.survey.nunique() > 1:
        raise ValueError("Multiple surveys found in item_df")
    if item_df.question.nunique() > 1:
        raise ValueError("Multiple items found in item_df")

    survey = item_df.survey.unique()[0]
    item = item_df.question.unique()[0]
    item_df = item_df.dropna(subset=["response_binary"])
    duration = item_df.duration.unique()[0]
    if item_df.duration.nunique() > 1:
        raise ValueError(
            f"Multiple durations found in item_df, {item_df.duration.unique()}"
        )
    item_df["survey_start"] = pd.to_datetime(item_df.survey_start)
    sensor_features["survey_start"] = pd.to_datetime(
        sensor_features.survey_start
    )

    use_sensor_features = sensor_features.merge(
        item_df[["user_id", "survey_start", "duration"]]
    )
    if use_sensor_features.QC_expected_duration.nunique() > 1:
        raise ValueError(
            f"Multiple durations found in sensor_features, {sensor_features.QC_expected_duration.unique()}"
        )

    if item_df.response_binary.nunique() < 2:
        print(
            f"Skipping {survey} - {item} due to homogenous response: {item_df.response_binary.value_counts()}"
        )
        return pd.DataFrame()

    if item_df.shape[0] < 20:
        print(
            f"Skipping {survey} - {item} due to low sample size ({item_df.shape[0]})"
        )
        return pd.DataFrame()
    if item_df.response.nunique() < 2:
        print(
            f"Skipping {survey} - {item} due to homogenous response: {item_df.response.value_counts()}"
        )
        return pd.DataFrame()

    item_folder = Path(output_folder, f"{survey}_{item}_{duration}")
    if not item_folder.exists():
        item_folder.mkdir()

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def getModelPredictions(input_args):
        model_name, model, param_grid = input_args
        model_out = Path(
            item_folder, f"{survey}_{item}_{model_name}_{duration}.pkl"
        )
        predictions_out = Path(
            item_folder,
            f"{survey}_{item}_{model_name}_{duration}_predictions.parquet",
        )
        if (
            model_out.exists()
            and predictions_out.exists()
            and not rerun_cached
        ):
            print(
                f"Skipping {survey} - {item} - {model_name} due to existing output"
            )
            return pd.read_parquet(predictions_out)
        predictions, trained_model = train_test_model(
            item_df,
            use_sensor_features,
            hk_features=feature_cols,
            demographics=None,
            demog_features=[],
            dropna_subset=nona_cols,
            inner_cv=inner_cv,
            model_name=model_name,
            model=model,
            param_grid=param_grid,
            n_folds=n_folds,
        )
        predictions["outcome"] = item
        predictions["survey"] = survey
        predictions["duration"] = duration
        predictions["model"] = model_name

        with open(model_out, "wb") as f:
            pickle.dump(trained_model, f)
        predictions.to_parquet(predictions_out, index=False)
        return predictions

    return pd.concat([getModelPredictions(args) for args in model_paramgrid])


## RUN MODELING
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--survey-data",
        type=str,
        help="csv file containing survey data processed with process_survey_data.py",
    )
    parser.add_argument(
        "--sensor-features",
        help="csv file containing sensor features",
    )
    parser.add_argument(
        "--binary-thresholds",
        type=str,
        help="csv file containing thresholds for binarizing survey responses",
    )
    parser.add_argument(
        "--output-folder", type=str, help="folder to write results to"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run in debug mode (only use 1 survey item)",
    )
    parser.add_argument(
        "--rerun-cached",
        action="store_true",
        help="rerun cached models",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="whether to run in parallel",
    )
    args = parser.parse_args()

    binary_metrics = ["roc_auc", "average_precision"]

    print("Checking input files")
    # Check that all files exist
    for item in [
        args.survey_data,
        args.sensor_features,
        args.binary_thresholds,
    ]:
        assert Path(item).exists(), f"{item} does not exist"

    print("Loading data")
    survey_data = pd.read_csv(args.survey_data, parse_dates=["survey_start"])
    sensor_features = pd.read_csv(
        args.sensor_features, parse_dates=["survey_start"]
    )
    binary_thresholds = pd.read_excel(args.binary_thresholds)

    survey_data = survey_data.merge(binary_thresholds, how="inner")
    survey_data["response_binary"] = (
        pd.to_numeric(survey_data.response) >= survey_data.threshold
    )
    print("setting duration for all PVSS & PHQ14 surveys to 7 days")
    survey_data.loc[
        survey_data.survey.isin(["pvss", "phq14"]),
        "duration",
    ] = "7days"

    if not Path(args.output_folder).exists():
        print(f"Creating output folder {args.output_folder}")
        Path(args.output_folder).mkdir()

    feature_cols = [
        c
        for c in sensor_features.columns
        if c
        not in [
            "user_id",
            "survey",
            "survey_start",
            "duration",
            "expected_duration",
        ]
        and not c.startswith("QC_")
    ]

    all_preds = []

    model_paramgrid = [
        (
            "rf",
            RandomForestClassifier(),
            {
                "model__n_estimators": [100, 200, 500],
                "model__max_depth": [5, 10, 20],
                "model__min_samples_leaf": [2, 5, 10],
                "model__max_features": [None, "sqrt", "log2"],
                "model__random_state": [42],
                "feature_select__k": [10, 50, "all"],
                "feature_select__score_func": [mutual_info_classif, f_classif],
            },
        ),
        (
            "gb",
            GradientBoostingClassifier(),
            {
                "model__n_estimators": [100, 200, 500],
                "model__max_depth": [5, 10, 20],
                "model__min_samples_leaf": [2, 5, 10],
                "model__max_features": [None, "sqrt", "log2"],
                "model__random_state": [42],
                "feature_select__k": [10, 50, "all"],
                "feature_select__score_func": [mutual_info_classif, f_classif],
            },
        ),
        (
            "lr",
            LogisticRegression(),
            {
                "feature_select__k": [10, 50, "all"],
                "feature_select__score_func": [mutual_info_classif, f_classif],
            },
        ),
        ("dummy", DummyClassifier(), {}),
    ]

    # Cross-validation performed for each item
    print("Running cross-validation")
    if args.debug:
        survey_data = survey_data[survey_data.question == "phq14_total_score"]
        args.parallel = False

    nona_cols = ["response_binary"]
    n_folds = 10
    rerun_cached = args.rerun_cached
    if args.parallel:
        print(
            f"Using {N_PROC} cores for parallel processing with {N_JOBS_CV} jobs per core"
        )
        all_preds = Parallel(n_jobs=N_PROC)(
            delayed(train_test_item_wrapper)(
                item_df,
                sensor_features,
                args.output_folder,
                feature_cols,
                model_paramgrid,
                nona_cols,
                n_folds,
                rerun_cached,
            )
            for _, item_df in tqdm(
                survey_data.groupby(["question", "survey"]),
                desc="Running cross-validation in parallel",
            )
        )
    else:
        all_preds = [
            train_test_item_wrapper(
                item_df,
                sensor_features,
                args.output_folder,
                feature_cols,
                model_paramgrid,
                nona_cols,
                n_folds,
                rerun_cached,
            )
            for (item, survey), item_df in tqdm(
                survey_data.groupby(["question", "survey"]),
                desc="Running cross-validation",
            )
        ]

    model_predictions = pd.concat(all_preds)
    not_feature_cols = [
        c for c in model_predictions.columns if c not in feature_cols
    ]
    model_predictions = model_predictions[not_feature_cols + feature_cols]
    print(
        "Saving predictions to",
        Path(args.output_folder, "binary_predictions.parquet"),
    )
    model_predictions.to_parquet(
        Path(args.output_folder, "binary_predictions.parquet"), index=False
    )
