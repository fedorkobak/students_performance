import pickle
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder

train = pd.read_parquet(Path()/".tmp"/"train.parquet")

ordinal_encoder = OrdinalEncoder().fit(train[[
    "event_name", "name"
]])
with open("src/ordinal_encoder.pkl", "wb") as f:
    pickle.dump(ordinal_encoder, f)
