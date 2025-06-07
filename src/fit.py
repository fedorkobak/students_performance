import pickle
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder

from src.data import SessionsDataSet


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
