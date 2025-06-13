import os
import tqdm
import torch
import dotenv
import mlflow
import logging

from src.model import BasicRNN
from src.data import get_loaders

loss_function = torch.nn.BCELoss()


def epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.utils.data.DataLoader
) -> list[float]:
    """
    Train model on the given data.

    Returns
    -------
    list[float]
        The values of the loss function for each batch.
    """
    losses = []
    model.train()
    model.requires_grad_(True)

    for X, y in tqdm.tqdm(data):
        optimizer.zero_grad()
        out = model(X)
        loss_value = loss_function(out, y)
        loss_value.backward()
        losses.append(float(loss_value))
        optimizer.step()
    return losses


def compute_loss(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader
) -> float:
    """
    Compute loss of the given model on the given data.
    """
    model = model.eval().requires_grad_(False)

    processed = [(model(X), y) for X, y in tqdm.tqdm(data)]
    p = torch.cat([v[0] for v in processed], dim=0)
    y = torch.cat([v[1] for v in processed], dim=0)
    return float(loss_function(p, y))


if __name__ == "__main__":
    dotenv.load_dotenv()
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT"])
    logging.basicConfig(level=logging.INFO)

    torch.manual_seed(1)
    torch.use_deterministic_algorithms(True)

    logging.info("Building data...")
    train_loader, test_loader = get_loaders(batch_size=32)
    dataset = train_loader.dataset.dataset

    model = BasicRNN(
        input_size=dataset.get_unit_shape(),
        hidden_size=15,
        output_size=18,
        num_layers=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    logging.info("Fitting model...")
    with mlflow.start_run():
        for e in range(3):
            loss_values = epoch(
                model=model,
                optimizer=optimizer,
                data=train_loader
            )
            mlflow.log_metric("BSE_train", loss_values[-1], step=(e + 1))

            test_loss = compute_loss(
                model=model,
                data=test_loader,
                step=(e + 1)
            )
            mlflow.log_metric("BSE_test", test_loss)

        mlflow.pytorch.log_model(model, "model")
        mlflow.sklearn.log_model(
            dataset.cat_features_encoder,
            "ordinal_encoder"
        )
