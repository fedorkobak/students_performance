import os
import torch
import dotenv
import mlflow
import logging
import argparse

from src.model import BasicRNN, TransformerEncoding
from src.data import get_datasets

from . import fit

parser = argparse.ArgumentParser(
    prog="ModelFit.",
    description="Fit the model."
)

subparsers = parser.add_subparsers(dest="model", required=True)
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=64)

rnn_parser = subparsers.add_parser("rnn")
rnn_parser.add_argument("--hidden-size", type=int, default=10)
rnn_parser.add_argument("--num-layers", type=int, default=1)

transformer_parser = subparsers.add_parser("transformer")
transformer_parser.add_argument("--dim-feedforward", type=int, default=256)
transformer_parser.add_argument("--nhead", type=int, default=1)
transformer_parser.add_argument("--num-layers", type=int, default=1)

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

    input_size = train_dataset.dataset.get_unit_shape()
    if args.model == "rnn":
        mlflow.log_params({
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
        })
        model = BasicRNN(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=18,
            num_layers=args.num_layers
        )
    elif args.model == "transformer":
        mlflow.log_params({
            "dim_feedforward": args.dim_feedforward,
            "nhead": args.nhead,
            "num_layers": args.num_layers
        })
        model = TransformerEncoding(
            d_model=input_size,
            dim_feedforward=args.dim_feedforward,
            nhead=args.nhead,
            num_layers=args.num_layers,
            output_size=18
        )

    logging.info("Fitting model...")
    fit(
        data_sets=(train_dataset, test_dataset),
        model=model,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        epochs=args.epochs
    )
