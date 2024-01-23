import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from predict_selfreport_items.predict_selfreport_CV import train_test_model


class TestPredictSelfReportCV(unittest.TestCase):
    def setUp(self):
        # Set up test data
        self.n = 100
        n = self.n
        self.outcome_df = pd.DataFrame(
            {
                "user_id": range(n),
                "survey_start": pd.date_range(
                    "2021-01-01", periods=n, freq="D"
                ),
                "response_binary": [1, 0] * (n // 2),
                "item": ["item1"] * n,
                "survey": ["survey1"] * n,
                "redcap_event_name": ["event1"] * n,
            }
        )
        self.hk_feature_df = pd.DataFrame(
            {
                "user_id": range(n),
                "survey_start": pd.date_range(
                    "2021-01-01", periods=n, freq="D"
                ),
                "feature1": range(n),
                "feature2": range(n),
                "feature3": range(n),
                "feature4": range(n),
                "feature5": range(n),
            }
        )
        self.metrics = ["roc_auc", "average_precision"]
        self.hk_features = [
            "feature1",
            "feature2",
            "feature3",
            "feature4",
            "feature5",
        ]
        self.demographics = None
        self.demog_features = []
        self.dropna_subset = ["feature1"]
        self.inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        self.model_name = "lr"
        self.model = LogisticRegression()
        self.param_grid = {
            "model__C": [0.1, 1, 10],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear"],
        }
        self.n_folds = 5

    # Basic test to make sure the function runs
    def test_train_test_model(self):
        # Test the train_test_model function
        result, model = train_test_model(
            self.outcome_df,
            self.hk_feature_df,
            self.hk_features,
            self.demographics,
            self.demog_features,
            self.dropna_subset,
            self.inner_cv,
            self.model_name,
            self.model,
            self.param_grid,
            self.n_folds,
        )
        assert result.shape[0] == self.n
        assert result.shape[1] == 14
        assert isinstance(model.named_steps["model"], LogisticRegression)


if __name__ == "__main__":
    unittest.main()
