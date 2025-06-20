# students_performance

Solution for the kaggle competition "[Predict Student Performance from Game Play](https://github.com/fedorkobak/students_performance.git)"

Check the mlflow server at `http://93.125.49.123:49998` for experiments. Use username and password from `.env` to login.

**The report describing the experiments** can be found in the ["analysis"](analysis.ipynb) notebook.

## Dev deploy

Python version `3.13.4`.

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

Run tests with `python -m unittest`.

## Training

Run the model fitting with `python3 -m src.fit`. To get help on the fit parameters, use `python3 -m src.fit --help`. Two models are implemented:

- Recurrent: `python -m src.fit rnn --help`.
- Transformer: `python -m src.fit transformer --help`.

For example, run the 10 epochs fitting process for the transformer model with two heads with command:

```bash
python3 -m src.fit --epochs 10 transformer --nhead 2`
```

## Run model

Run the model using:

```bash
python3 -m src.run <file_name>
```

The `<file_name>` must refer to a `.csv` file that contains a sequence of events - for example, [`template.csv`](template.csv).

The result would be 18 float numbers, each corresponding to an estimate of the probability that the user will answer the question correctly.

```bash
$ python3 -m src.run template.csv
0.6647812128067017 0.9790564775466919 0.937534511089325 0.7482591867446899 0.44910088181495667 0.6920568346977234 0.6698082089424133 0.5708855390548706 0.6778101325035095 0.41098877787590027 0.570005476474762 0.8471113443374634 0.1813270002603531 0.66145259141922 0.4007180631160736 0.7202093005180359 0.6435446739196777 0.9471038579940796
```
