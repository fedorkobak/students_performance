import os
import dotenv
import mlflow
import logging
import mlflow.pytorch
import mlflow.sklearn

from src.model import BaiscRNN
# from src.data import train_dataset, train_loader, ordinal_encoder

dotenv.load_dotenv()
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT"])
logging.basicConfig(level=logging.INFO)

unit_size = train_dataset[0][0].shape[1]
model = BaiscRNN(
    input_size=unit_size,
    hidden_size=unit_size*2,
    output_size=18,
    num_layers=1
)

with mlflow.start_run():
    mlflow.pytorch.log_model(model)
    mlflow.sklearn.log_model(ordinal_encoder, "ordinal_encoder")
