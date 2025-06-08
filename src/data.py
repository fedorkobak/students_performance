"""Data primitives"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import torch
import torch.utils
from torch.utils.data import Dataset


class SessionsDataSet(Dataset):
    """
    The dataset that takes a table of sessions and, in each iteration, returns
    a matrix of prepared data for a single session.

    Parameters
    ----------
    df: pd.DataFrame
        Must have "session_id" and "elapsed_time" columns.
    labels: pd.DataFrame
        Must have "session_id", "question_id" and "correct" columns.
    raw_features: list[str]
        A set of features that must not be transformed.
    cat_features_encoder: OrdinalEncoder
        Object that transforms categorial features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        raw_features: list[str],
        cat_features_encoder: OrdinalEncoder
    ):

        if set(labels["session_id"]) != set(df["session_id"]):
            raise ValueError(
                "`df` and `lables` must have the same elements set of "
                "`session_id`"
            )
        if "session_id" not in df.columns:
            raise ValueError("`session_id` wasn't found in the `df`")
        if "session_id" not in labels.columns:
            raise ValueError("`session_id` wasn't found in the `labels`")
        if "elapsed_time" not in df.columns:
            raise ValueError("`elapsed_time` wasn't found in the `df`")
        if "question_id" not in labels.columns:
            raise ValueError("`question_id` wasn't found in the `labels`")
        if "correct" not in labels.columns:
            raise ValueError("`correct` wasn't found in the `labels`")

        super().__init__()
        self.df = df
        self.labels = labels
        self.cat_features_encoder = cat_features_encoder
        self.raw_features = raw_features

        self.sessions_ids = df["session_id"].unique()

    def __len__(self) -> int:
        return len(self.sessions_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get one session info.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - The events of the sessions in order according to the increasing
            of the "elapsed_time".
            - A 1d vector of labels is sorted in ascending order by question.
        """

        my_id = self.sessions_ids[index]

        sessions_for_id = (
            self.df.loc[self.df["session_id"] == my_id]
            .sort_values("elapsed_time")
        )
        raw_features = torch.tensor(sessions_for_id[self.raw_features].values)
        cat_features = torch.tensor(self.cat_features_encoder.transform(
            sessions_for_id[self.cat_features_encoder.feature_names_in_]
        ))
        X = torch.concat([raw_features, cat_features], axis=1)

        y = torch.tensor(
            self.labels.loc[self.labels["session_id"] == my_id]
            .sort_values("question_id")
            .loc[:, "correct"]
            .values,
            dtype=torch.float32
        )

        return X, y

    def transform_categorial(self, df: pd.DataFrame) -> torch.Tensor:
        features = self.cat_features_encoder.feature_names_in_
        return torch.tensor(
            self.cat_features_encoder.transform(df[features])
        )


def collate_sessions(
    units: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate the units of the `SessionDataset`.

    Parameters
    ----------
    units: list[tuple[torch.Tensor, torch.Tensor]]
        List of items of the `SessionDataSet`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - Sessions padded to the max session and concatenated. So the
        dimentinality of the output is:
            - len of units (batch size).
            - max seqence len.
            - size of the vector that represents the action.
        - Answers to the qestions collated to the batch.
    """
    X = torch.nn.utils.rnn.pad_sequence(
        [u[0] for u in units], batch_first=True
    )
    y = torch.stack([u[1] for u in units])
    return X, y
