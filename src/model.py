import torch
import torch.nn as nn


class BasicRNN(nn.Module):
    """
    Basic implementation of the RNN model.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers
        )
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        rnn_out = self.rnn(X)[-1][-1]
        linear_out = self.linear(rnn_out)
        return torch.sigmoid(linear_out)
