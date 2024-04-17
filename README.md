# LSTM Models for SciCite Citation Classification

This repository contains code for LSTM-based models for SciCite citation classification. The repository consists of the following files:

- `config.py`: Configuration file containing hyperparameters and model settings.
- `dataloader.py`: Data loader module for loading and preprocessing the SciCite dataset.
- `lstm.py`: Implementation of LSTM models including LSTM, BiLSTM, LSTM with self-attention, and BiLSTM with self-attention.
- `train.py`: Script for training the LSTM models.
- `test.py`: Script for evaluating trained models on test data.

## Usage

1. Set hyperparameters and model configurations in `config.py`.
2. Run `train.py` to train the models using the specified configurations.
3. After training, use `test.py` to evaluate the trained models on test data.

Make sure to have the necessary dependencies installed as specified in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
