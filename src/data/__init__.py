import pandas as pd
import multiprocessing
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


def process_model_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """
    Initial processing of information.
    """
    # Counting a delay of the event
    data["event_delay"] = (
        data["elapsed_time"] - data["elapsed_time"].shift(1)
    ).fillna(0)

    data[raw_features] = data[raw_features].fillna(-1)
    return data


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and prepares data for use in subsequent steps.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Input data for the model.
        - Labels of the model.
    """
    data = process_model_inputs(pd.read_parquet(Path()/".tmp"/"train.parquet"))
    labels = pd.read_parquet(Path()/".tmp"/"train_labels.parquet")

    return data, labels


def get_datasets() -> tuple[td.Dataset, td.Dataset]:
    """
    Get test and train datasets.
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
    return train_dataset, test_dataset


def get_loaders(
    batch_size: int,
    num_workers: int = multiprocessing.cpu_count(),
    datasets: tuple[td.Dataset, td.Dataset] | None = None
) -> tuple[td.DataLoader, td.DataLoader]:
    """
    Get train and test data loaders.

    Parameters
    ----------
    batch_size: int
        Number of the observations in the single batch.
    num_workers: int
        Number of workers that can be used by the loaders.
    datasets: tuple[td.Dataset, td.Dataset] | None
        Train and test datasets.
        If `None`, `get_datasets` will be used to get datasets.
    """
    if datasets is None:
        train_dataset, test_dataset = get_datasets()
    else:
        train_dataset, test_dataset = datasets

    train_loader = td.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_sessions,
        num_workers=num_workers
    )
    test_loader = td.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_sessions,
        num_workers=num_workers
    )
    return train_loader, test_loader
