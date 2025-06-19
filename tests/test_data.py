"""Tests for data primitives"""
import torch
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.data.utils import (
    SessionsDataSet, collate_sessions, sessions_df_to_torch
)

import unittest
from unittest import TestCase

pd.set_option('future.no_silent_downcasting', True)


class TestSessionsDfToTorch(TestCase):
    """
    Test src.data.utils.sessions_df_to_torch function.
    """
    session_df = pd.DataFrame({
        "a": ["a", "a", "b"],
        "b": [1, 2, 3]
    })

    encoder_mock = unittest.mock.Mock()
    encoder_mock.feature_names_in_ = ["a"]
    encoder_mock.transform = (
        lambda df: df[["a"]].replace({"a": 1, "b": 2}).values.astype(float)
    )

    def test(self):
        ans = sessions_df_to_torch(
            df=self.session_df,
            raw_features=["b"],
            cat_features_encoder=self.encoder_mock
        )
        exp = torch.tensor([[1, 1], [2, 1], [3, 2]], dtype=torch.float32)
        torch.testing.assert_close(ans, exp)


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

        raw = torch.tensor([1, 3, 2], dtype=torch.float32)[:, None]
        subset = (
            self
            .test_data.loc[self.test_data["session_id"] == 1]
            .sort_values("elapsed_time")
        )
        cat = torch.tensor(
            self.encoder.transform(subset[["a"]]),
            dtype=torch.float32
        )
        exp_X = torch.concat([raw, cat], axis=1)
        torch.testing.assert_close(ans_X, exp_X)

        exp_y = torch.tensor(
            [0, 0, 1], dtype=torch.float32
        )
        torch.testing.assert_close(ans_y, exp_y)

    def test_unit_shape(self):
        self.assertEqual(self.sessions_data_set.get_unit_shape(), 2)


class TestCollateSessions(TestCase):
    def test_collation(self):
        sequences = [
            torch.tensor([
                [1, 2, 3, 4],
                [5, 3, 2, 7]
            ]),
            torch.tensor([
                [3, 3, 3, 1]
            ])
        ]
        y = [
            torch.tensor([1, 0, 1]),
            torch.tensor([0, 0, 0])
        ]
        inp = [(sequences[0], y[0]), (sequences[1], y[1])]

        ans = collate_sessions(inp)
        exp = (
            torch.tensor([
                [
                    [1, 2, 3, 4],
                    [5, 3, 2, 7]
                ],
                [
                    [3, 3, 3, 1],
                    [0, 0, 0, 0]
                ]
            ]),
            torch.tensor([
                [1, 0, 1],
                [0, 0, 0]
            ])
        )

        torch.testing.assert_close(ans[0], exp[0])
        torch.testing.assert_close(ans[1], exp[1])
