"""Tests for data primitives"""
import torch
import pandas as pd
from unittest import TestCase
from sklearn.preprocessing import OrdinalEncoder

from src.data import SessionsDataSet


class TestSessionsDataSet(TestCase):
    test_data = pd.DataFrame({
        "session_id": [1, 1, 1, 2, 2, 2],
        "a": ["a", "b", "c", "a", "b", "c"],
        "b": [1, 2, 3, 4, 5, 6]
    })
    encoder = OrdinalEncoder().fit(test_data[["a"]])
    sessions_data_set = SessionsDataSet(
        df=test_data,
        raw_features=["b"],
        cat_features_encoder=encoder
    )

    def test_item(self):
        ans = self.sessions_data_set[0]

        subset = self.test_data.loc[self.test_data["session_id"] == 1]
        raw = torch.tensor(subset[["b"]].values)
        cat = torch.tensor(self.encoder.transform(subset[["a"]]))
        exp = torch.concat([raw, cat], axis=1)

        torch.testing.assert_close(ans, exp)
