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
    """

    cat_features_encoder: OrdinalEncoder | None = None

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        if self.cat_features_encoder is None:
            raise ValueError("`cat_features_encoder` doesn't defined.")
        self.sessions_ids = df["session_id"].unique()

    def __len__(self) -> int:
        return len(self.sessions_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        return super().__getitem__(index)

    @classmethod
    def transform_categorial(cls, df: pd.DataFrame) -> torch.Tensor:
        features = cls.cat_features_encoder.feature_names_in_
        return torch.tensor(
            cls.cat_features_encoder.transform(df[features])
        )
