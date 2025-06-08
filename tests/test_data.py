"""Tests for data primitives"""
import torch
import pandas as pd
from unittest import TestCase
from sklearn.preprocessing import OrdinalEncoder

from src.data import SessionsDataSet


class TestSessionsDataSet(TestCase):
    test_data = pd.DataFrame({
        "session_id": [1, 1, 1, 2, 2, 2],
        "elapsed_time": [1, 3, 2, 4, 5, 6],
        "a": ["a", "b", "c", "a", "b", "c"],
        "b": [1, 2, 3, 4, 5, 6]
    })
    test_labels = pd.DataFrame({
        "session_id": [1, 1, 1, 2, 2, 2],
        "question_id": [1, 3, 2, 1, 2, 3],
        "correct": [0, 1, 0, 1, 0, 1]
    })
    encoder = OrdinalEncoder().fit(test_data[["a"]])
    sessions_data_set = SessionsDataSet(
        df=test_data,
        labels=test_labels,
        raw_features=["b"],
        cat_features_encoder=encoder
    )

    def test_item(self):
        ans_X, ans_y = self.sessions_data_set[0]

        raw = torch.tensor([1, 3, 2])[:, None]
        subset = (
            self
            .test_data.loc[self.test_data["session_id"] == 1]
            .sort_values("elapsed_time")
        )
        cat = torch.tensor(self.encoder.transform(subset[["a"]]))
        exp_X = torch.concat([raw, cat], axis=1)
        torch.testing.assert_close(ans_X, exp_X)

        exp_y = torch.tensor(
            [0, 0, 1], dtype=torch.float32
        )
        torch.testing.assert_close(ans_y, exp_y)
