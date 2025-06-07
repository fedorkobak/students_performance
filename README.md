# students_performance
Solution for the kaggle competition "[Predict Student Performance from Game Play](https://github.com/fedorkobak/students_performance.git)"

## Dev deploy

To deploy development envrionment use: 

- Install all necessary packages in your environment `pip isntall -e .`.
    - **Note**: This project uses PyTorch. Specify  the [build](https://pytorch.org/get-started/locally/) to be installed with the `--extra-index-url` parameter. Use `pip3 install --extra-index-url https://download.pytorch.org/whl/cpu -e` for the most basic CPU installation.
- Put the archive with training data in the `.tmp` folder and run `python3 procedure.py`.
