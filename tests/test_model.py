"""
Set tests for model
"""
import torch
from unittest import TestCase
from model import BasicRNN, TransformerEncoding


class TestOut(TestCase):
    """
    Check the properties of the output of the models.
    """
    torch.random.manual_seed(1)
    # parameters of the tensors
    # supposed that the input would be a batch with <batch_size> seqeunces of
    # <seqence_size> elements, each of which would be <input_size>-dimensional
    # elements.
    batch_size = 4
    sequence_size = 10
    input_size = 15
    output_size = 18
    # input example that represents all
    input_example = torch.rand(batch_size, sequence_size, input_size)

    # outputs of models which properties have to be checked
    outputs = {
        "basic_rnn": BasicRNN(
            input_size=input_size,
            hidden_size=5,
            output_size=output_size,
            num_layers=1
        )(input_example),
        "transformer_encoding": TransformerEncoding(
            d_model=input_size,
            dim_feedforward=1,
            nhead=1,
            num_layers=1,
            output_size=output_size
        )(input_example)
    }

    def test_shape(self):
        """
        Is the shape of the model's ouputs correct?
        """
        for name, out in self.outputs.items():
            self.assertEqual(
                out.shape,
                torch.Size((
                    self.batch_size, self.output_size
                )),
                msg=f"Shape for {name} is wrong!"
            )

    def test_proba_propertie(self):
        """
        If the data have a probability properties.
        """
        for name, out in self.outputs.items():
            self.assertTrue(
                bool((out > 0).all()),
                msg=(
                    "The probability cannot be less than zero. "
                    f"But it occures in {name}."
                )
            )
            self.assertTrue(
                bool((out < 1).all()),
                msg=(
                    "The probability cannot be more than one."
                    f"But it occures in {name}."
                )
            )
