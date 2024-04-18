# XLnet for SciCite Citation Classification

This repository contains code for to train a Xlnet classifier for SciCite citation classification and provide some interpretable html plots. The repository consists of the following files:

- `config.py`: Configuration file model settings and save path
- `dataloaders.py`: Creates a dataloader for train and val/test
- `dataset.py`: Dataset reader to create dataset
- `tokenize.py`: Function to tokenize the dataset via Xlnet Tokenizer
- `train_xlnet.py`: Script for training the XLnet.
- `test_xlnet.py`: Script for evaluating trained Xlnet on test data.
- `model_interpretability.py`: Script to interpret model using lime

## Usage

1. Set hyperparameters and file configurations in `config.py`.
2. Run `train_xlnet.py` to train the models using the specified configurations.
3. Save the model into a model directory and update `config.py` pretrained_dir parameter
3. After training, use `test_xlnet.py` to evaluate the trained models on test data.
4. Run `model_interpretability.py` to evaluate model

Make sure to have the necessary dependencies installed as specified in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
