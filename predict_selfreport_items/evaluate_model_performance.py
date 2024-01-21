# Description: Evaluate model performance for predicting self-report items

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
import numpy as np
from joblib import Parallel, delayed
import pingouin as pg
import matplotlib.pyplot as plt
import shap

def try_roc_auc(x):
    try:
        return roc_auc_score(x.y_true, x.y_pred_proba)
    except ValueError:
        return np.nan


def try_average_precision(x):
    try:
        return average_precision_score(x.y_true, x.y_pred_proba)
    except ValueError:
        return np.nan


def try_balanced_accuracy(x):
    try:
        return balanced_accuracy_score(x.y_true, x.y_pred)
    except ValueError:
        return np.nan


def calc_performance(
    prediction_df: pd.DataFrame,
    groupby: list[str] = ["model", "outcome", "fold"],
):
    roc_auc = prediction_df.groupby(groupby).apply(try_roc_auc)
    pr_auc = prediction_df.groupby(groupby).apply(try_average_precision)
    balanced_accuracy = prediction_df.groupby(groupby).apply(
        try_balanced_accuracy
    )
    n = prediction_df.groupby(groupby).size()
    sample_performance = pd.concat(
        [roc_auc, pr_auc, balanced_accuracy, n], axis=1
    )
    sample_performance.columns = [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "n",
    ]
    return sample_performance


def get_bootstrapped_performance(
    prediction_df: pd.DataFrame,
    groupby: list[str] = ["model", "survey", "outcome", "fold"],
    unit: str = "user_id",
    n_boot: int = 100,
):
    def bootstrap_sample_perf(i):
        bootstrap_sample = prediction_df.groupby(groupby + [unit]).sample(
            n=1, replace=True
        )
        sample_perf = calc_performance(bootstrap_sample, groupby=groupby)
        sample_perf["has_null"] = sample_perf.isnull().any(axis=1)
        sample_perf["bootstrap"] = i
        return sample_perf

    bootstrap_performance = pd.concat(
        Parallel(n_jobs=10, verbose=10)(
            delayed(bootstrap_sample_perf)(i) for i in range(n_boot)
        )
    )

    # To do modify to make bootstrap a column that counts succesfful bootstraps
    mean_bootstrapped_perf_df = (
        bootstrap_performance.reset_index()
        .groupby(groupby)
        .aggregate(
            {
                "has_null": "sum",
                "roc_auc": "mean",
                "pr_auc": "mean",
                "balanced_accuracy": "mean",
            }
        )
        .reset_index()
        .rename(
            columns={
                "roc_auc": "test_roc_auc",
                "pr_auc": "test_average_precision",
            }
        )
    )
    mean_bootstrapped_perf_df["n_bootstraps"] = n_boot
    return mean_bootstrapped_perf_df


def get_best_model_performance_and_predictions(
    performance_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    metric: str = "test_roc_auc",
    groupby: list[str] = ["outcome"],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Group by outcome, feature_set, and model, and calculate the mean of test_roc_auc
    grouped_df = performance_df.groupby(groupby + ['model'])[
        metric
    ].mean()

    # Find the model with the highest test_roc_auc for each outcome and feature_set
    best_models = (
        grouped_df.groupby(groupby)
        .idxmax()
        .str[-1]
        .reset_index()
    )
    best_models = best_models.rename(columns={metric: "model"})

    # Merge the best_models with the original dataframe to get the full row
    best_model_performance = best_models.merge(
        performance_df.reset_index(), how="left", validate="1:m"
    )

    # Get the predictions for the best model
    best_model_predictions = predictions_df.merge(
        best_model_performance, how="inner", validate="m:1"
    )

    return best_model_performance, best_model_predictions, best_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model performance for predicting self-report items"
    )
    parser.add_argument(
        "--output_folder",
        default="results",
        help="Folder to save results to",
    )
    parser.add_argument(
        "--predictions_file",
        help='Path to predictions file from predict_selfreport_evaluation.py.',
        required=True,
    )

    args = parser.parse_args()
    model_predictions = pd.read_csv(args.predictions_file)

    binary_performance_df = get_bootstrapped_performance(
        model_predictions,
        groupby=["model", "survey", "outcome", "fold"],
        unit="user_id",
        n_boot=100,
    )

    (
        best_model_performance,
        best_model_predictions,
        best_models,
    ) = get_best_model_performance_and_predictions(
        binary_performance_df,
        model_predictions,
        metric="test_roc_auc",
    )
    best_model_performance["p"] = np.nan
    binary_perf_results = []
    for it, it_df in best_model_performance.groupby(
        ["survey", "outcome", "model"]
    ):
        # test that test_roc_auc is > 0 with p value of 0.05 or less
        auroc_result = pg.ttest(it_df.test_roc_auc, 0.5, alternative="greater")
        auroc_result["survey"] = it[0]
        auroc_result["outcome"] = it[1]
        auroc_result["model"] = it[2]
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
                "outcome",
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
    # TODO: Refactor looking at SHAP values to another file
    sig_perf = binary_performance_test[
        binary_performance_test.p_adj < 0.05
    ].copy()
    significant_surveys = binary_performance_test[
        binary_performance_test.outcome.isin(sig_perf.outcome.tolist())
    ][["survey", "outcome"]]

    for (survey, item), s_df in best_model_predictions[
        best_model_predictions.outcome.isin(significant_surveys.outcome.unique())
    ].groupby(["survey", "outcome"]):
        shap.summary_plot(
            np.stack(s_df.SHAP_values.values),
            s_df[feature_cols],
            show=False,
            max_display=10,
            plot_size=(10, 5),
        )
        item_map = {}
        survey_map = {}
        if item in item_map.keys():
            title = f"{survey_map[survey]}: {item_map[item]}"
        else:
            title = f"{survey}: {item}"
        plt.title(title, fontsize=15)
        plt.show()
