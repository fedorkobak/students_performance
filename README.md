# students_performance

Solution for the kaggle competition "[Predict Student Performance from Game Play](https://github.com/fedorkobak/students_performance.git)"

Check the mlflow server at `http://93.125.49.123:49998` for experiments. Use username and password from `.env` to login.

**The report describing the experiments** can be found in the ["analysis"](analysis.ipynb) notebook.

## Dev deploy

To deploy development envrionment use:

- Install all necessary packages in your environment `pip isntall -e .`.
  - **Note**: This project uses PyTorch. Specify  the [build](https://pytorch.org/get-started/locally/) to be installed with the `--extra-index-url` parameter. Use `pip3 install --extra-index-url https://download.pytorch.org/whl/cpu -e .` for the most basic CPU installation.
- Put the archive with training data in the `.tmp` folder and run `python3 procedure.py`.
- Create `.env` file with following content:

```bash
MLFLOW_TRACKING_URI="http://93.125.49.123:49998"
MLFLOW_TRACKING_USERNAME="IFORTEX"
MLFLOW_TRACKING_PASSWORD="ifortex_cred"
MLFLOW_EXPERIMENT="ifortex_test_task"
```

## Run

Run the model fitting with `python3 -m src.fit`. To get help on the fit parameters, use `python3 -m src.fit --help`. Two models are implemented:

- Recurrent: `python -m src.fit rnn --help`.
- Transformer: `python -m src.fit transformer --help`.

For example, run the 10 epochs fitting process for the transformer model with two heads with command:

`python3 -m src.fit --epochs 10 transformer --nhead 2`

Run tests with `python -m unittest`.
