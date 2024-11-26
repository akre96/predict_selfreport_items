# Description: Evaluate model performance for predicting self-report items

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
)
import numpy as np
from joblib import Parallel, delayed
import pingouin as pg
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm


SURVEY_ITEM_MAP = {
    "phq14_total_score": "PHQ-8 Total Score",
    "phq_1": "Little interest or pleasure in doing things",
    "phq_10": "Trouble concentrating on things",
    "phq_11": "Moving or speaking slowly",
    "phq_12": "Being fidgety or restless",
    "phq_13": "Feeling irritable",
    "phq_14": "Little interest in sex",
    "phq_2": "Feeling down, depressed",
    "phq_3": "Feeling hopeless",
    "phq_4": "Trouble falling asleep or staying asleep",
    "phq_5": "Sleeping too much",
    "phq_6": "Feeling tired or having little energy",
    "phq_7": "Poor appetite",
    "phq_8": "Overeating",
    "phq_9": "Feeling bad about yourself",
    "psqi_bathroom": "Bathroom use",
    "psqi_bedtime": "Bedtime",
    "psqi_cannot_breathe": "Cannot breathe comfortably",
    "psqi_cannot_sleep_30mins": "Cannot get to sleep within 30 minutes",
    "psqi_cold": "Feel too cold",
    "psqi_cough": "Cough or snore loudly",
    "psqi_daydys": "Daytime dysfunction due to sleepiness",
    "psqi_distb": "Sleep disturbance",
    "psqi_enthusiasm": "Keep up enough enthusiasm",
    "psqi_hot": "Feel too hot",
    "psqi_hours_sleep": "Hours of sleep",
    "psqi_hse_tmphse": "Sleep Efficiency",
    "psqi_laten": "Sleep Latency",
    "psqi_meds_sleep": "Medication for sleep",
    "psqi_nightmares": "Nightmares",
    "psqi_othr_reasons": "Other reasons",
    "psqi_pain": "Have pain",
    "psqi_rate_sleep": "Sleep rating",
    "psqi_sleep_dur": "Duration of sleep",
    "psqi_time_fall_asleep": "Minutes to fall asleep",
    "psqi_total": "PSQI total score",
    "psqi_trouble_awake": "Trouble staying awake",
    "psqi_wakeup": "Wakeup time",
    "psqi_wakeup_night": "Wake up at night",
    "pvss_1": "Savored food",
    "pvss_10": "Praise about work",
    "pvss_11": "Time with others",
    "pvss_12": "Accomplish goals",
    "pvss_13": "Hugged by loved one",
    "pvss_14": "Activity with friends",
    "pvss_15": "Positive feedback on projects",
    "pvss_16": "Upcoming meal",
    "pvss_17": "Reached a goal",
    "pvss_18": "Hug made happy after parted",
    "pvss_19": "Expect to master task",
    "pvss_20": "Pursued fun activities",
    "pvss_21": "Admired beauty around me",
    "pvss_2": "Energy in to activity I enjoy",
    "pvss_3": "Fresh air outdoors",
    "pvss_4": "Time with people I know",
    "pvss_5": "Weekend activity sustained good mood",
    "pvss_6": "Physical contact with person I felt close to",
    "pvss_7": "Expected to enjoy moment outdoors",
    "pvss_8": "Looked forward to feedback on work",
    "pvss_9": "Expected to enjoy meals",
    "pvss_effort_val_domain": "Effort valuation domain",
    "pvss_food_total": "Food score",
    "pvss_goals_total": "Goals score",
    "pvss_hobbies_total": "Hobbies score",
    "pvss_outdoors_total": "Outdoors score",
    "pvss_physical_total": "Physical touch score",
    "pvss_posfeed_total": "Positive feedvack score",
    "pvss_response_domain": "Initial responsiveness domain",
    "pvss_rew_anticip_domain": "Reward anticipation domain",
    "pvss_rew_expect_domain": "Reward expectancy domain",
    "pvss_rew_val_domain": "Reward valuation domain",
    "pvss_satiation_domain": "Reward satiation domain",
    "pvss_social_total": "Social interaction score",
    "pvss_total": "PVSS total score",
}


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


# calculate sensitivity and specificity
def try_sensitivity(x):
    try:
        return recall_score(x.y_true, x.y_pred)
    except ValueError:
        return np.nan


def try_specificity(x):
    try:
        tn, fp, fn, tp = confusion_matrix(x.y_true, x.y_pred).ravel()
        return tn / (tn + fp)
    except ValueError:
        return np.nan


def calc_performance(
    prediction_df: pd.DataFrame,
    groupby: list[str] = ["model", "outcome", "fold"],
):
    # Filter groups with only one class
    prediction_filtered_df = prediction_df.groupby(groupby).filter(
        lambda x: x.y_true.nunique() > 1
    )
    if prediction_filtered_df.empty:
        return pd.DataFrame()

    roc_auc = prediction_filtered_df.groupby(groupby).apply(
        try_roc_auc, include_groups=False
    )
    pr_auc = prediction_filtered_df.groupby(groupby).apply(
        try_average_precision, include_groups=False
    )
    balanced_accuracy = prediction_filtered_df.groupby(groupby).apply(
        try_balanced_accuracy,
        include_groups=False,
    )
    sensitivity = prediction_filtered_df.groupby(groupby).apply(
        try_sensitivity,
        include_groups=False,
    )
    specificity = prediction_filtered_df.groupby(groupby).apply(
        try_specificity,
        include_groups=False,
    )
    n = prediction_df.groupby(groupby).size()
    sample_performance = pd.concat(
        [roc_auc, pr_auc, balanced_accuracy, n, sensitivity, specificity],
        axis=1,
    )
    sample_performance.columns = [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "n",
        "sensitivity",
        "specificity",
    ]
    return sample_performance


def get_bootstrapped_performance(
    prediction_df: pd.DataFrame,
    groupby: list[str] = ["model", "survey", "outcome", "fold"],
    unit: str = "user_id",
    n_boot: int = 100,
):
    def get_bootstrap_sample(i):
        bootstrap_sample = prediction_df.groupby(groupby + [unit]).sample(
            n=1, replace=True, random_state=i
        )
        return bootstrap_sample

    bootstrap_sample = pd.concat(
        Parallel(n_jobs=10, verbose=10)(
            delayed(get_bootstrap_sample)(i) for i in range(n_boot)
        )
    )
    bootstrap_performance = calc_performance(
        bootstrap_sample, groupby=groupby
    ).rename(
        columns={
            "roc_auc": "test_roc_auc",
            "pr_auc": "test_average_precision",
            "balanced_accuracy": "test_balanced_accuracy",
        }
    )

    # To do modify to make bootstrap a column that counts succesfful bootstraps
    bootstrap_performance["n_bootstraps"] = n_boot
    return bootstrap_performance.reset_index()


def get_best_model_performance_and_predictions(
    performance_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    metric: str = "test_roc_auc",
    groupby: list[str] = ["outcome"],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Group by outcome, feature_set, and model, and calculate the mean of test_roc_auc
    grouped_df = performance_df.groupby(groupby + ["model"])[metric].mean()

    # Find the model with the highest test_roc_auc for each outcome and feature_set
    best_models = grouped_df.groupby(groupby).idxmax().str[-1].reset_index()
    best_models = best_models.rename(columns={metric: "model"})

    # Merge the best_models with the original dataframe to get the full row
    best_model_performance = best_models.merge(
        performance_df.reset_index(), how="left", validate="1:m"
    )

    # Get the predictions for the best model
    best_model_predictions = predictions_df.merge(
        best_model_performance[
            [
                *groupby,
                "model",
                "test_roc_auc",
                "test_average_precision",
                "fold",
            ]
        ],
        how="inner",
        validate="m:1",
    )

    return best_model_performance, best_model_predictions, best_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model performance for predicting self-report items"
    )
    parser.add_argument(
        "--output_folder",
        default="tmp_out",
        help="Folder to save results to",
    )
    parser.add_argument(
        "--predictions_file",
        default="tmp_out/binary_predictions.parquet",
        help="Path to predictions file from predict_selfreport_evaluation.py.",
    )

    args = parser.parse_args()
    model_predictions = pd.read_parquet(args.predictions_file)

    # Print sample size used
    sample_size_inp = model_predictions[
        ["user_id", "redcap_event_name", "survey"]
    ].drop_duplicates()
    print(
        "\n",
        "Sample size:",
        "\n",
        sample_size_inp.groupby("survey")
        .aggregate({"user_id": "nunique", "redcap_event_name": "count"})
        .rename(
            columns={"user_id": "n_users", "redcap_event_name": "n_responses"}
        ),
        "\n",
    )

    info_cols = [
        "y_true",
        "y_pred",
        "y_pred_proba",
        "SHAP_values",
        "redcap_event_name",
        "survey_start",
        "user_id",
        "fold",
        "model",
        "outcome",
        "survey",
        "n_users",
        "n",
        "duration",
    ]
    feature_cols = [c for c in model_predictions.columns if not c in info_cols]

    # Calculate performance
    print("Calculating performance\n")
    outpath = Path(args.output_folder, "binary_performance.csv")
    if outpath.exists():
        print("Loading performance from", outpath)
        binary_performance_df = pd.read_csv(outpath)
    else:
        binary_performance_df = get_bootstrapped_performance(
            model_predictions,
            groupby=["model", "survey", "duration", "outcome", "fold"],
            unit="user_id",
            n_boot=100,
        )
        binary_performance_df.to_csv(outpath, index=False)

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
    for it, it_df in tqdm(
        best_model_performance.groupby(
            ["survey", "outcome", "model", "duration"]
        ),
        desc="Testing performance",
    ):
        # test that test_roc_auc is > 0 with p value of 0.05 or less
        auroc_result = pg.ttest(it_df.test_roc_auc, 0.5, alternative="greater")
        auroc_result["survey"] = it[0]
        auroc_result["outcome"] = it[1]
        auroc_result["model"] = it[2]
        auroc_result["duration"] = it[3]
        auroc_result["mean_auroc"] = it_df.test_roc_auc.mean()
        auroc_result["median_auroc"] = it_df.test_roc_auc.median()
        auroc_result["max_auroc"] = it_df.test_roc_auc.max()
        auroc_result["min_auroc"] = it_df.test_roc_auc.min()
        auroc_result["normality"] = pg.normality(it_df.test_roc_auc)[
            "normal"
        ].iloc[0]
        binary_perf_results.append(auroc_result)

    binary_performance_test = pd.concat(binary_perf_results)

    def adjust_pval(group):
        group["p_adj"] = pg.multicomp(group["p-val"], method="fdr_bh")[1]
        return group

    binary_performance_test = binary_performance_test.groupby("survey").apply(
        adjust_pval
    )
    binary_performance_test["label"] = binary_performance_test.outcome.map(
        SURVEY_ITEM_MAP
    )
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
                "label",
                "model",
                "duration",
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

    ## Plot SHAP values for significant outcomes
    sig_perf = binary_performance_test[
        binary_performance_test.p_adj < 0.5
    ].copy()
    significant_surveys = binary_performance_test[
        binary_performance_test.outcome.isin(sig_perf.outcome.tolist())
    ][["survey", "outcome", "label"]]

    for (survey, item, duration), s_df in best_model_predictions.groupby(
        ["survey", "outcome", "duration"]
    ):
        # fill missing values with median per feature
        s_df[feature_cols] = s_df[feature_cols].fillna(
            s_df[feature_cols].median()
        )
        shap_vals = np.stack(s_df.SHAP_values.values)
        # Bound shap values to 99.9th percentile
        shap_vals = np.where(
            shap_vals > np.percentile(shap_vals, 99.9),
            np.percentile(shap_vals, 99.9),
            shap_vals,
        )
        shap_vals = np.where(
            shap_vals < np.percentile(shap_vals, 0.1),
            np.percentile(shap_vals, 0.1),
            shap_vals,
        )
        shap.summary_plot(
            shap_vals,
            s_df[feature_cols],
            show=False,
            max_display=10,
            plot_size=(10, 5),
        )
        item_map = {}
        survey_map = {}
        if item in SURVEY_ITEM_MAP.keys():
            title = f"{survey}: {SURVEY_ITEM_MAP[item]}"
        else:
            title = f"{survey}: {item}"
        plt.title(title, fontsize=15)
        shap_out_folder = Path(
            args.output_folder,
            f"{survey}_{item}_{duration}",
        )
        if not shap_out_folder.exists():
            shap_out_folder.mkdir(parents=True)
        # Save plot
        plt.savefig(
            Path(
                shap_out_folder,
                f"shap_{survey}_{item}.png",
            ),
            bbox_inches="tight",
        )
        plt.close()
