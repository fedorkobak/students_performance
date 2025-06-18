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


class TransformerEncoding(nn.Module):
    """
    This is a model that is supposed to use a transformer-like encoder.

    Parameters
    ----------
    d_model: int
        Dimensionality of the model. It is actually the dimentionality of the
        one unit of the sequence that is supposed to be processed.
    dim_feedforward: int
        Dimentionality of the feed forward layer of transformer.
    nhead: int
        Number of transformer heads.
    num_layers: int
        Number of transformers layers to be stacked.
    max_seq_length: int
        Maximum length of the sequence that will be processed.
        All later sequence elements will be deblocked in forward.
    """
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        num_layers: int,
        output_size: int,
        max_seq_length: int | None = None
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dim_feedforward=dim_feedforward
            ),
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.linear = nn.Linear(
            in_features=d_model,
            out_features=output_size
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = (
            X[..., :self.max_seq_length, :]
            if self.max_seq_length is not None
            else X
        )
        encoded = self.transformer_encoder(X)
        logits = self.linear(encoded.mean(dim=-2))
        return torch.sigmoid(logits)
