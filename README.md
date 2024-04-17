# cs4248

This repository contains the code for a text classification project which uses Convolutional Neural Networks (CNN) and shallow neural network. 

## Project Structure

- `preprocess.ipynb`: Jupyter notebook containing preprocessing steps for the dataset.
- `Shallow NN.ipynb`: Jupyter notebook detailing the implementation and training of a shallow neural network model.
- `CNN+sentecepiece`: Python script for creating a CNN model combined with SentencePiece tokenization and train & test part.
- `activations_heatmap.py`: Python script for visualizing activation heatmaps, help to understand which parts of the input are activating certain neurons.
- The next five are all about TextCNN without SentencePiece tokenization
- `TextCNN.py`: Python script containing the definition of the TextCNN model architecture.
- `TextNumericalization.py`: Python script for numericalizing text data into a format suitable for model training.
- `test.py`: Script for testing the model on a dataset, generating performance metrics like accuracy, precision, recall, and F1 score.
- `train.py`: Main script used to train the text classification model, including loading data, setting up the model, and initiating the training loop.
- `utils.py`: Utility functions that might be used across the project for various tasks like data loading, transformation, etc.
