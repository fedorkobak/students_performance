import os
import tqdm
import torch
import dotenv
import mlflow
import logging

from src.model import BasicRNN
from src.data import get_loaders


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
    loss = torch.nn.BCELoss()

    for X, y in tqdm.tqdm(data):
        optimizer.zero_grad()
        out = model(X)
        loss_value = loss(out, y)
        loss_value.backward()
        losses.append(float(loss_value))
        optimizer.step()
    return losses


if __name__ == "__main__":
    dotenv.load_dotenv()
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT"])
    logging.basicConfig(level=logging.INFO)

    train_loader, test_loader = get_loaders()
    basic_rnn = BasicRNN(
        input_size=train_loader.dataset.dataset.get_unit_shape(),
        hidden_size=15,
        output_size=train_loader.dataset.dataset.labels["question_id"].nunique(),
        num_layers=1
    )
    # with mlflow.start_run():
    #     mlflow.pytorch.log_model(model)
    #     mlflow.sklearn.log_model(ordinal_encoder, "ordinal_encoder")
