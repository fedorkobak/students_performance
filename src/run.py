import os
import torch
import dotenv
import mlflow
import argparse
import pandas as pd

from src.data import raw_features, process_model_inputs
from src.data.utils import sessions_df_to_torch

dotenv.load_dotenv()
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = 'false'

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("file", type=str)
args = argument_parser.parse_args()

cat_features_encoder = mlflow.sklearn.load_model(
    "models:/students-performance-cat-encoder/latest"
)
predict_model: torch.nn.Module = mlflow.pytorch.load_model(
    "models:/students-performance-model/latest"
)
predict_model.eval().requires_grad_(False)

data = sessions_df_to_torch(
    df=process_model_inputs(pd.read_csv(args.file)),
    raw_features=raw_features,
    cat_features_encoder=cat_features_encoder
)

print(" ".join(map(lambda n: str(float(n)), predict_model(data))))
