"""
Set tests for model
"""
import torch
from unittest import TestCase
from model import BasicRNN, TransformerEncoding

torch.random.manual_seed(1)


class TestModels(TestCase):
    '''
    This is a class that is supposed to be a parent class. It simply defines
    instances of the models that have to be tested in child classes.

    Attributes
    ----------
    input_size: int
        Dimentionality of the one element of the sequences taht will be
        processed by the models.
    ouput_size: int
        Dementinality of the ouput of the models.
    '''
    input_size = 15
    output_size = 18
    models = {
        "basic_rnn": BasicRNN(
            input_size=input_size,
            hidden_size=5,
            output_size=output_size,
            num_layers=1
        ),
        "transformer_encoding": TransformerEncoding(
            d_model=input_size,
            dim_feedforward=1,
            nhead=1,
            num_layers=1,
            output_size=output_size
        )
    }


class TestOutProperties(TestModels):
    """
    Check the properties of the output of the models.

    Parameters
    ----------
    batch_size: int
        Size of the batch of the example input.
    sequence_size: int
        Size seqences size of the example input.
    input_example: torch.tensor
        Example input.
    """
    # parameters of the tensors
    # supposed that the input would be a batch with <batch_size> seqeunces of
    # <seqence_size> elements, each of which would be <input_size>-dimensional
    # elements.
    batch_size = 4
    sequence_size = 10
    # input example that represents all
    input_example = torch.rand(
        batch_size,
        sequence_size,
        TestModels.input_size
    )

    # outputs of models which properties have to be checked
    outputs = {}
    for name, model in TestModels.models.items():
        outputs[name] = model(input_example)

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
