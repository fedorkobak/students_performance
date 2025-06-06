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

logging.info("Converting to parquet...")
pd.read_csv(csv_path/"train.csv").to_parquet(tmp_path/"train.parquet")
pd.read_csv(
    csv_path/"train_labels.csv"
).to_parquet(
    tmp_path/"train_labels.parquet"
)
