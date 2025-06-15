import os
import torch
import dotenv
import mlflow
import logging
import argparse

from src.model import BasicRNN
from src.data import get_datasets

from . import fit


parser = argparse.ArgumentParser(
    prog="ModelFit.",
    description="Fit the model."
)
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--hidden-size", type=int, default=10)
parser.add_argument("--num-layers", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=30)
args = parser.parse_args()

dotenv.load_dotenv()
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT"])
logging.basicConfig(level=logging.INFO)

torch.manual_seed(1)
torch.use_deterministic_algorithms(True)
torch.multiprocessing.set_sharing_strategy("file_system")

logging.info("Loading data...")
train_dataset, test_dataset = get_datasets()


with mlflow.start_run(run_name=args.run_name):
    mlflow.log_params({
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
    })

    model = BasicRNN(
        input_size=train_dataset.dataset.get_unit_shape(),
        hidden_size=args.hidden_size,
        output_size=18,
        num_layers=args.num_layers
    )

    logging.info("Fitting model...")
    fit(
        data_sets=(train_dataset, test_dataset),
        model=model,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        epochs=args.epochs
    )
