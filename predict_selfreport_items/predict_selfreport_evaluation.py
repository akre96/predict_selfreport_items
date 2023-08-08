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

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
import shap
import argparse
import pandas as pd
import numpy as np
import pingouin as pg
from pathlib import Path
import pickle
from tqdm import tqdm

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
    args = parser.parse_args()

    # Check that all files exist
    for item in [
        args.survey_data,
        args.sensor_features,
        args.binary_thresholds,
    ]:
        assert Path(item).exists(), f"{item} does not exist"

    survey_data = pd.read_csv(args.survey_data, parse_dates=["survey_start"])
    sensor_features = pd.read_csv(
        args.sensor_features, parse_dates=["survey_start"]
    )
    binary_thresholds = pd.read_excel(args.binary_thresholds).dropna()

    survey_data = survey_data.merge(binary_thresholds, how="inner")
    survey_data["response_binary"] = (
        pd.to_numeric(survey_data.response) > survey_data.threshold_leq
    )
    sensor_survey = sensor_features.merge(
        survey_data, how="inner", validate="1:m"
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
    performances = []
    used_data = []

    cv = StratifiedGroupKFold(n_splits=10)

    # Cross-validation performed for each item
    for (item, survey), item_df in tqdm(
        sensor_survey.groupby(["question", "survey"]),
    ):
        for f in feature_cols:
            item_df[f] = item_df[f].replace([np.inf, -np.inf], np.nan)
            item_df[f] = item_df[f].astype("float32")

        item_df = item_df.dropna(subset=["response_binary", *feature_cols])
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
        # Fit model on all data and save result as pickle
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(item_df[feature_cols], item_df["response_binary"])
        out_path = Path(args.output_folder, f"{survey}_{item}_model.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(clf, f)

        # Perform cross-validation
        clf = GradientBoostingClassifier(random_state=42)
        try:
            performance = pd.DataFrame(
                cross_validate(
                    clf,
                    item_df[feature_cols],
                    item_df["response_binary"],
                    cv=cv,
                    groups=item_df["user_id"],
                    scoring=[
                        "average_precision",
                        "roc_auc",
                    ],
                    return_estimator=True,
                    n_jobs=-1,
                )
            )

            ix_training, ix_test = [], []
            # Loop through each fold and append the training & test indices to the empty lists above
            for fold in cv.split(
                item_df[feature_cols],
                item_df["response_binary"],
                item_df["user_id"],
            ):
                ix_training.append(fold[0])
                ix_test.append(fold[1])

            SHAP_values_per_fold = []

            # Loop through each outer fold and extract SHAP values
            responses = []
            predictions = []
            uids = []
            for i, (train_outer_ix, test_outer_ix) in enumerate(
                zip(ix_training, ix_test)
            ):  # -#-#
                X_train, X_test = (
                    item_df.iloc[train_outer_ix][feature_cols],
                    item_df.iloc[test_outer_ix][feature_cols],
                )
                y_train, y_test = (
                    item_df["response_binary"].iloc[train_outer_ix],
                    item_df["response_binary"].iloc[test_outer_ix],
                )
                model = performance["estimator"][i]
                y_hat = model.predict_proba(X_test)
                responses.append(y_test)
                predictions.append(y_hat)
                uids.append(item_df.iloc[test_outer_ix]["user_id"])

                # Use SHAP to explain predictions
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                SHAP_values_per_fold.append(shap_values)  # -#-#
        except ValueError as e:
            print(f"Skipping {survey} - {item} due to error: {e}")
            continue
        performance["survey"] = survey
        performance["item"] = item
        performance["fold"] = performance.index
        performance["n"] = item_df.shape[0]
        performance["n_users"] = item_df["user_id"].nunique()
        performance["SHAP_values"] = SHAP_values_per_fold
        performance["ix_test"] = ix_test
        performance["responses"] = responses
        performance["predictions"] = predictions
        performance["uids"] = uids
        performances.append(performance)
        used_data.append(item_df)

    binary_performance_df = pd.concat(performances)
    print(
        "Saving performance results to",
        Path(args.output_folder, "binary_performance.csv"),
    )
    binary_performance_df.to_csv(
        Path(args.output_folder, "binary_performance.csv"), index=False
    )
    # Get feature importances
    feature_importances = []
    for i, row in binary_performance_df.iterrows():
        feats = pd.DataFrame({"features": feature_cols})
        feats["gini_impurity"] = row["estimator"].feature_importances_
        feats["survey"] = row.survey
        feats["item"] = row["item"]
        feats["fold"] = row.fold
        feats["n"] = row.n
        feats["n_users"] = row.n_users

        feature_importances.append(feats)

    print(
        "Saving feature importances to",
        Path(args.output_folder, "gini_feature_importances.csv"),
    )
    binary_fi_df = pd.concat(feature_importances)
    binary_fi_df.to_csv(
        Path(args.output_folder, "gini_feature_importances.csv"), index=False
    )

    # Test that performance > 0.5 ROC AUC
    binary_performance_df["p"] = np.nan
    binary_perf_results = []
    for it, it_df in binary_performance_df.groupby(["survey", "item"]):
        # test that test_roc_auc is > 0 with p value of 0.05 or less
        auroc_result = pg.ttest(it_df.test_roc_auc, 0.5, alternative="greater")
        auroc_result["survey"] = it[0]
        auroc_result["item"] = it[1]
        auroc_result["mean_auroc"] = it_df.test_roc_auc.mean()
        auroc_result["median_auroc"] = it_df.test_roc_auc.median()
        auroc_result["max_auroc"] = it_df.test_roc_auc.max()
        auroc_result["min_auroc"] = it_df.test_roc_auc.min()
        auroc_result["normality"] = pg.normality(it_df.test_roc_auc)[
            "normal"
        ].iloc[0]
        binary_perf_results.append(auroc_result)

    binary_performance_test = pd.concat(binary_perf_results)
    binary_performance_test["p_adj"] = pg.multicomp(
        binary_performance_test["p-val"], method="fdr_bh"
    )[1]
    binary_performance_test = binary_performance_test.sort_values(
        "p_adj", ascending=True
    )
    print(
        "Saving performance test results to",
        Path(args.output_folder, "binary_performance_test.csv"),
    )
    binary_performance_test.to_csv(
        Path(args.output_folder, "binary_performance_test.csv"), index=False
    )
    print("Top Models:")
    print(
        binary_performance_test.sort_values(by="p_adj")[
            [
                "survey",
                "item",
                "mean_auroc",
                "median_auroc",
                "max_auroc",
                "min_auroc",
                "normality",
                "p-val",
                "p_adj",
            ]
        ]
    )

    # SHAP feature importance -- not saved yet
    """
    significant_surveys = binary_performance_test[
        binary_performance_test.item.isin(sig_perf.item.tolist())
    ][["survey", "item"]]

    for (survey, item), s_df in binary_performance_df[
        binary_performance_df.item.isin(significant_surveys.item.unique())
    ].groupby(["survey", "item"]):
        new_index = [
            ix for ix_test_fold in s_df["ix_test"].to_numpy() for ix in ix_test_fold
        ]
        item_df = (
            sensor_survey.loc[
                (sensor_survey.survey == survey) & (sensor_survey.question == item)
            ]
            .dropna(subset=["response_binary", *feature_cols])
            .copy()
            .reset_index(drop=True)
        )
        SHAP_values_per_fold = []
        for ar in s_df["SHAP_values"].to_numpy():
            for val in ar:
                SHAP_values_per_fold.append(val)
        shap.summary_plot(
            np.array(SHAP_values_per_fold),
            item_df.reindex(new_index)[feature_cols],
            show=False,
            max_display=10,
            plot_size=(10, 5),
        )
        if item in item_map.keys():
            title = f"{survey_map[survey]}: {item_map[item]}"
        else:
            title = f"{survey}: {item}"
        plt.title(title, fontsize=15)
        plt.show()
    """
