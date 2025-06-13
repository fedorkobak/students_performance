import os
import tqdm
import dotenv
import mlflow
import logging

from src.model import BasicRNN
from src.data import get_loaders, get_datasets

import torch
import torch.utils.data as td

loss_function = torch.nn.BCELoss()


def epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.utils.data.DataLoader,
    tqdm_desc: str | None = None
) -> list[float]:
    """
    Train model on the given data.

    Parameters
    ----------
    model: torch.nn.Module,
        Model to be trained.
    optimizer: torch.optim.Optimizer,
        The optimizer must be used to train the model.
    data: torch.utils.data.DataLoader,
        Data is needed to train the model.
    tqdm_desc: str | None
        Message for progress bar.

    Returns
    -------
    list[float]
        The values of the loss function for each batch.
    """
    losses = []
    model.train()
    model.requires_grad_(True)

    for X, y in tqdm.tqdm(data, desc=tqdm_desc):
        optimizer.zero_grad()
        out = model(X)
        loss_value = loss_function(out, y)
        loss_value.backward()
        losses.append(float(loss_value))
        optimizer.step()
    return losses


def compute_loss(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    tqdm_desc: str | None = None
) -> float:
    """
    Compute loss of the given model on the given data.
    model: torch.nn.Module,
        Model to be evaluated.
    data: torch.utils.data.DataLoader,
        The data on which the model must be evaluated.
    tqdm_desc: str | None = None
        Message for progress bar.
    """
    model = model.eval().requires_grad_(False)

    processed = [(model(X), y) for X, y in tqdm.tqdm(data, desc=tqdm_desc)]
    p = torch.cat([v[0] for v in processed], dim=0)
    y = torch.cat([v[1] for v in processed], dim=0)
    return float(loss_function(p, y))


def fit(
    data_sets: tuple[td.Dataset, td.Dataset],
    model: torch.nn.Module,
    batch_size: int,
    lr: float,
    epochs: int
):
    """
    Fit model.

    Parameters
    ----------
    data_sets: tuple[td.Dataset, td.Dataset],
        Datasets that are supposed to be used for model building.
    model: torch.nn.Module,
        Model to fit.
    batch_size: int,
        Batch size.
    lr: float,
        Learning rate.
    epochs: int
        Number of epochs.
    """

    train_loader, test_loader = get_loaders(
        datasets=data_sets,
        batch_size=batch_size
    )
    dataset = train_loader.dataset.dataset

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run():
        for e in range(epochs):
            epoch(
                model=model,
                optimizer=optimizer,
                data=train_loader,
                tqdm_desc="Training model"
            )

            test_loss = compute_loss(
                model=model,
                data=test_loader,
                tqdm_desc="Evaluating on test"
            )
            train_loss = compute_loss(
                model=model,
                data=train_loader,
                tqdm_desc="Evaluating on train"
            )
            mlflow.log_metric("BCE_test", test_loss, step=(e + 1))
            mlflow.log_metric("BCE_train", train_loss, step=(e + 1))

        mlflow.pytorch.log_model(model, "model")
        mlflow.sklearn.log_model(
            dataset.cat_features_encoder,
            "ordinal_encoder"
        )


if __name__ == "__main__":
    dotenv.load_dotenv()
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT"])
    logging.basicConfig(level=logging.INFO)

    torch.manual_seed(1)
    torch.use_deterministic_algorithms(True)

    logging.info("Loading data...")
    train_dataset, test_dataset = get_datasets()

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
        epochs=5
    )
