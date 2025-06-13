import os
import torch
import dotenv
import logging
import mlflow

from src.model import BasicRNN
from src.data import get_datasets

from . import fit

dotenv.load_dotenv()
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT"])
logging.basicConfig(level=logging.INFO)

torch.manual_seed(1)
torch.use_deterministic_algorithms(True)

logging.info("Loading data...")
train_dataset, test_dataset = get_datasets()


with mlflow.start_run():
    model = BasicRNN(
        input_size=train_dataset.dataset.get_unit_shape(),
        hidden_size=15,
        output_size=18,
        num_layers=1
    )

    logging.info("Fitting model...")
    fit(
        data_sets=(train_dataset, test_dataset),
        model=model,
        batch_size=64,
        lr=1e-4,
        epochs=1
    )
