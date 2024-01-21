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
import multiprocessing

NUM_CORES = multiprocessing.cpu_count()

def train_test_model(
    outcome_df: pd.DataFrame,
    hk_feature_df: pd.DataFrame,
    metrics: list[str],
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
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, object]:
    ml_data = hk_feature_df.merge(
        outcome_df, on=["user_id", "survey_start"], how="left", validate="1:1"
    ).dropna(subset=dropna_subset)

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
            ("imputer", SimpleImputer(strategy="median")),
            ("variance_threshold", VarianceThreshold()),
            ("feature_select", SelectKBest(k="all")),
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
        n_jobs=5,
    )
    cv = StratifiedGroupKFold(n_splits=n_folds)

    # Get individual fold prediction errors, split by weeks since baseline
    preds = []
    for i, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=ml_data.user_id)
    ):
        random_search.fit(X.iloc[train_idx], y.iloc[train_idx])
        m = random_search.best_estimator_.named_steps['model']

        if isinstance(m, DummyClassifier):
            shap_vals = np.zeros((len(test_idx), len(input_features)))
            selected_feature_names = input_features
        else:
            preprocess = Pipeline(random_search.best_estimator_.steps[:-1])
            # Get the feature selection transformer
            feature_select = random_search.best_estimator_.named_steps['feature_select']

            # Get the boolean mask of selected features
            selected_mask = feature_select.get_support()

            # Get the feature names from the imputer
            imputer = random_search.best_estimator_.named_steps['imputer']
            feature_names_after_imputation = imputer.get_feature_names_out()

            # Index into the feature names with the mask
            selected_feature_names = feature_names_after_imputation[selected_mask]
            if isinstance(m, RandomForestClassifier) or isinstance(m, GradientBoostingClassifier):
                explainer = shap.TreeExplainer(m)
            elif isinstance(m, LogisticRegression):
                explainer = shap.LinearExplainer(m, preprocess.transform(X.iloc[train_idx]))
            else:
                raise InvalidModelError(f"Model {m} not supported")
            raw_shap_vals = explainer.shap_values(preprocess.transform(X.iloc[test_idx]))

            temp_shap_df = pd.DataFrame(raw_shap_vals, columns=selected_feature_names)
            shap_df = temp_shap_df.reindex(columns=input_features, fill_value=0)
            shap_vals = shap_df.to_numpy()


        
        preds.append(
            pd.DataFrame(
                {
                    "y_true": y.iloc[test_idx],
                    "y_pred": random_search.predict(X.iloc[test_idx]),
                    "y_pred_proba": random_search.predict_proba(X.iloc[test_idx])[
                        :, 1
                    ],
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
        type=str,
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
    args = parser.parse_args()

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    binary_metrics = ["roc_auc", "average_precision"]

    print('Checking input files')
    # Check that all files exist
    for item in [
        args.survey_data,
        args.sensor_features,
        args.binary_thresholds,
    ]:
        assert Path(item).exists(), f"{item} does not exist"
    
    print('Loading data')
    survey_data = pd.read_csv(args.survey_data, parse_dates=["survey_start"])
    sensor_features = pd.read_csv(
        args.sensor_features, parse_dates=["survey_start"]
    )
    binary_thresholds = pd.read_excel(args.binary_thresholds).dropna()

    survey_data = survey_data.merge(binary_thresholds, how="inner")
    survey_data["response_binary"] = (
        pd.to_numeric(survey_data.response) > survey_data.threshold_leq
    )

    if not Path(args.output_folder).exists():
        print(f"Creating output folder {args.output_folder}")
        Path(args.output_folder).mkdir()

    feature_cols = [
        c
        for c in sensor_features.columns
        if c not in ["user_id", "survey", "survey_start", "duration"]
        and not c.startswith("QC_")
    ]

    used_data = []
    all_preds = []

    cv = StratifiedGroupKFold(n_splits=10)
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
                "feature_select__k": [10, "all"],
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
                "feature_select__k": [10, "all"],
                "feature_select__score_func": [mutual_info_classif, f_classif],
            },
        ),
        (
            "lr",
            LogisticRegression(),
            {
                "feature_select__k": [10, "all"],
                "feature_select__score_func": [mutual_info_classif, f_classif],
            },
        ),
        ("dummy", DummyClassifier(), {}),
    ]
    if args.debug:
        model_paramgrid = [model_paramgrid[2]]

    # Cross-validation performed for each item
    print("Running cross-validation")
    debug_run_complete = False
    for i, ((item, survey), item_df) in enumerate(tqdm(
        survey_data.groupby(["question", "survey"]),
    )):
        if args.debug and debug_run_complete:
            break

        item_df = item_df.dropna(subset=["response_binary"])
        if item_df.response_binary.nunique() < 2:
            print(
                f"Skipping {survey} - {item} due to homogenous response: {item_df.response_binary.value_counts()}"
            )
            continue

        if item_df.shape[0] < 20:
            print(
                f"Skipping {survey} - {item} due to low sample size ({item_df.shape[0]})"
            )
            continue
        if item_df.response.nunique() < 2:
            print(
                f"Skipping {survey} - {item} due to homogenous response: {item_df.response.value_counts()}"
            )
            continue

        for model_name, model, param_grid in tqdm(model_paramgrid, leave=False, desc=f"Training Models {survey} - {item}"):
            predictions, trained_model = train_test_model(
                item_df,
                sensor_features,
                binary_metrics,
                feature_cols,
                demographics=None,
                demog_features=[],
                dropna_subset=["response_binary"],
                inner_cv=inner_cv,
                model_name=model_name,
                model=model,
                param_grid=param_grid,
                n_folds=5,
            )
            predictions["outcome"] = item
            predictions["survey"] = survey
            predictions["model"] = model_name
            predictions["n_users"] = item_df.user_id.nunique()
            predictions["n"] = item_df.shape[0]
            all_preds.append(predictions)
            used_data.append(item_df)

            model_folder = Path(args.output_folder, f"{survey}_{item}")
            if not model_folder.exists():
                model_folder.mkdir()
            out_path = Path(
                model_folder, f"{survey}_{item}_{model_name}.pkl"
            )
            with open(out_path, "wb") as f:
                pickle.dump(trained_model, f)
        debug_run_complete = True

    model_predictions = pd.concat(all_preds)
    not_feature_cols = [c for c in model_predictions.columns if c not in feature_cols]
    model_predictions = model_predictions[not_feature_cols + feature_cols]
    print('Saving predictions to', Path(args.output_folder, "binary_predictions.csv"))
    model_predictions.to_csv(Path(args.output_folder, "binary_predictions.csv"), index=False)
