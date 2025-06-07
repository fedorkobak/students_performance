"""Data primitives"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import torch
from torch.utils.data import Dataset


def data_to_tenor(data: pd.DataFrame) -> torch.Tensor:
    pass


class SessionsDataSet(Dataset):
    """
    The dataset that takes a table of sessions and, in each iteration, returns
    a matrix of prepared data for a single session.

    Parameters
    ----------
    df: pd.DataFrame
    raw_features: list[str]
        A set of features that must not be transformed.
    cat_features_encoder: OrdinalEncoder
        Object that transforms categorial features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        raw_features: list[str],
        cat_features_encoder: OrdinalEncoder
    ):
        super().__init__()
        self.df = df
        self.cat_features_encoder = cat_features_encoder
        self.raw_features = raw_features

        self.sessions_ids = df["session_id"].unique()

    def __len__(self) -> int:
        return len(self.sessions_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        sessions_for_id = self.df.loc[
            self.df['session_id'] == self.sessions_ids[index]
        ]
        raw_features = torch.tensor(sessions_for_id[self.raw_features].values)
        cat_features = torch.tensor(self.cat_features_encoder.transform(
            sessions_for_id[self.cat_features_encoder.feature_names_in_]
        ))
        return torch.concat([raw_features, cat_features], axis=1)

    def transform_categorial(self, df: pd.DataFrame) -> torch.Tensor:
        features = self.cat_features_encoder.feature_names_in_
        return torch.tensor(
            self.cat_features_encoder.transform(df[features])
        )
