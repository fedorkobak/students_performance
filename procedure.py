import logging
import pandas as pd
from pathlib import Path
from zipfile import ZipFile

logging.basicConfig(level=logging.INFO)

# loading the csv takes a long time, it will be converted to the parquet
tmp_path = Path()/".tmp"
archive_path = tmp_path/"predict-student-performance-from-game-play.zip"
csv_path = tmp_path/"csv_data"
csv_path.mkdir(exist_ok=True)

logging.info("Unzipping...")
with ZipFile(file=archive_path, mode="r") as f:
    f.extractall(csv_path)

logging.info("Loading csv...")
train = pd.read_csv(csv_path/"train.csv")
train_labels = pd.read_csv(csv_path/"train_labels.csv")

logging.info("Preprocessing...")
train_labels[["session_id", "question_id"]] = (
    train_labels["session_id"].str.extract(r"(\d+)_q(\d+)")
    .apply(lambda col: col.astype("int"))
)
train_labels = train_labels[
    ['session_id', 'question_id', 'correct', ]
]

logging.info("Converting to parquet...")
train.to_parquet(tmp_path/"train.parquet")
train_labels.to_parquet(tmp_path/"train_labels.parquet")
