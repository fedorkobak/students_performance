import pickle
import logging
import pandas as pd
from pathlib import Path

import torch
from sklearn.preprocessing import OrdinalEncoder
import torch.utils

from src.data import SessionsDataSet

logging.basicConfig(level=logging.INFO)

logging.info("Processing data...")
train = pd.read_parquet(Path()/".tmp"/"train.parquet")

# Counting a delay of the event
train["event_delay"] = (
    train["elapsed_time"] - train["elapsed_time"].shift(1)
).fillna(0)
train["hover_duration"].fillna(-1)


ordinal_encoder = OrdinalEncoder().fit(train[[
    "event_name",
    "name",
    "text",
    "fqid",
    "room_fqid",
    "text_fqid",
    "level_group"
]])
with open("src/ordinal_encoder.pkl", "wb") as f:
    pickle.dump(ordinal_encoder, f)

logging.info("Constructing dataset...")

sessions_dataset = SessionsDataSet(
    df=train,
    raw_features=[
        "event_delay",
        "level",
        "room_coor_x",
        "room_coor_y",
        "screen_coor_x",
        "screen_coor_y",
        "hover_duration"
    ],
    cat_features_encoder=ordinal_encoder
)

train, test = torch.utils.data.random_split(
    sessions_dataset,
    [0.8, 0.2],
    generator=torch.Generator().manual_seed(1)
)
