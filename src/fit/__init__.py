import tqdm
import mlflow
import typing

from src.data import get_loaders

import torch
import torch.utils.data as td

from sklearn.metrics import roc_auc_score

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
        The optimizer.
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


def evaluate(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    tqdm_desc: str | None = None
) -> tuple[float, float]:
    """
    Computes the BSE loss and ROC AUC of the model on given model.

    Parameters
    ----------
    model: torch.nn.Module,
        Model to be evaluated.
    data: torch.utils.data.DataLoader,
        The data on which the model must be evaluated.
    tqdm_desc: str | None = None
        Message for progress bar.

    Returns
    -------
    tuple[float, float]
        BCE loss and ROC AUC.
    """
    model = model.eval().requires_grad_(False)

    processed = [(model(X), y) for X, y in tqdm.tqdm(data, desc=tqdm_desc)]
    p = torch.cat([v[0] for v in processed], dim=0)
    y = torch.cat([v[1] for v in processed], dim=0)
    return float(loss_function(p, y)), roc_auc_score(y_true=y, y_score=y)


def train_loop(
    epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loaders: tuple[td.DataLoader, td.DataLoader]
) -> typing.Generator[tuple[int, float, float, float, float]]:
    """
    Generator each step of wich is an epoch of model training with evaluation.

    Parameters
    ----------
    model: torch.nn.Module,
        Model to be trained.
    optimizer: torch.optim.Optimizer,
        The optimizer.
    loaders: tuple[td.DataLoader, td.DataLoader]
        Train and test loaders accordingly.

    Returns
    -------
    typing.Generator[tuple[int, float, float, float, float]]
        - Epoch number.
        - Train BCE.
        - Train ROC AUC.
        - Test BCE.
        - Test ROC AUC.
    """
    train_loader, test_loader = loaders
    for e in range(epochs):
        epoch(
            model=model,
            optimizer=optimizer,
            data=train_loader,
            tqdm_desc="Training model"
        )
        test_loss, test_auc = evaluate(
            model=model,
            data=test_loader,
            tqdm_desc="Evaluating on test"
        )
        train_loss, train_auc = evaluate(
            model=model,
            data=train_loader,
            tqdm_desc="Evaluating on train"
        )

        yield e, train_loss, train_auc, test_loss, test_auc


def fit(
    data_sets: tuple[td.Dataset, td.Dataset],
    model: torch.nn.Module,
    batch_size: int,
    lr: float,
    epochs: int
):
    """
    Fits model and logs fitting process to mlflow.
    Note: supposed to be used in the mlflow run context.

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
    if mlflow.active_run() is None:
        raise ValueError("Mlflow context didn't found.")

    train_loader, test_loader = get_loaders(
        datasets=data_sets,
        batch_size=batch_size
    )
    dataset = train_loader.dataset.dataset

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mlflow.log_params({
        "batch_size": batch_size,
        "lr": lr
    })

    t_loop = train_loop(
        epochs=epochs,
        model=model,
        optimizer=optimizer,
        loaders=(train_loader, test_loader)
    )
    for e, train_loss, train_auc, test_loss, test_auc in t_loop:
        mlflow.log_metrics(
            {
                "BCE_test": test_loss,
                "BCE_train": train_loss,
                "AUC_test": test_auc,
                "AUC_train": train_auc
            },
            step=(e + 1)
        )

    mlflow.pytorch.log_model(model, "model")
    mlflow.sklearn.log_model(
        dataset.cat_features_encoder,
        "ordinal_encoder"
    )
