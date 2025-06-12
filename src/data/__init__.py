import pandas as pd
from pathlib import Path

import torch
import torch.utils.data as td
from sklearn.preprocessing import OrdinalEncoder

from src.data.utils import SessionsDataSet, collate_sessions

raw_features = [
    "event_delay",
    "level",
    "page",
    "room_coor_x",
    "room_coor_y",
    "screen_coor_x",
    "screen_coor_y",
    "hover_duration"
]
encoded_features = [
    "event_name",
    "name",
    "text",
    "fqid",
    "room_fqid",
    "text_fqid",
    "level_group"
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and prepares data for use in subsequent steps.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Input data for the model.
        - Labels of the model.
    """
    train = pd.read_parquet(Path()/".tmp"/"train.parquet")
    train_labels = pd.read_parquet(Path()/".tmp"/"train_labels.parquet")

    # Counting a delay of the event
    train["event_delay"] = (
        train["elapsed_time"] - train["elapsed_time"].shift(1)
    ).fillna(0)
    train["hover_duration"].fillna(-1)

    train[raw_features] = train[raw_features].fillna(-1)
    return train, train_labels


def get_loaders() -> tuple[td.DataLoader, td.DataLoader]:
    """
    Get train and test data loaders.
    """
    data, labels = load_data()

    ordinal_encoder = OrdinalEncoder().fit(data[encoded_features])

    sessions_dataset = SessionsDataSet(
        df=data,
        labels=labels,
        raw_features=raw_features,
        cat_features_encoder=ordinal_encoder
    )

    train_dataset, test_dataset = td.random_split(
        sessions_dataset,
        [0.8, 0.2],
        generator=torch.Generator().manual_seed(1)
    )

    batch_size = 32
    train_loader = td.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_sessions
    )
    test_loader = td.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_sessions
    )
    return train_loader, test_loader
