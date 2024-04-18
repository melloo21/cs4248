import os
import time
import datetime
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from config import *
from dataset import CreateDataset
from tokenize import CreateTokens
from dataloaders import CreateDataloaders
from transformers import get_linear_schedule_with_warmup
from transformers import XLNetForSequenceClassification, AdamW, XLNetConfig
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

xlnet_loaded = AutoModelForSequenceClassification.from_pretrained(pretrained_dir)
xlnet_tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)

# https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/sentiment_analysis/Emotion%20classification%20multiclass%20example.html
xlnet_loaded.to(device)

# Reading all dataset
Scicite_data_getter = CreateDataset(filepath=FILEPATH)

test_label = Scicite_data_getter._label_to_id("test")
test_sentence = Scicite_data_getter._get_sentence_data("test")

test_input_ids, test_attention_masks, test_labels = tokenize_utils.xlnettokenize(
    all_sentences=test_sentence,labels=test_label)
prediction_dataloader = dataloader_utils.get_eval_dataloader(
    input_ids=test_input_ids, attention_masks=test_attention_masks, labels=test_labels)

# Put xlnet_loaded in evaluation mode
xlnet_loaded.eval()

# Tracking variables
predictions , true_labels = [], []

# Predict
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)

  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch

  with torch.no_grad():
      outputs = xlnet_loaded(b_input_ids, token_type_ids=None,
                      attention_mask=b_input_mask)

  logits = outputs.logits

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()

  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

# Calculate accuracy and F1 score
predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels)
pred_labels = np.argmax(predictions, axis=1)
acc = accuracy_score(true_labels, pred_labels)
f1_micro = f1_score(true_labels, pred_labels, average='micro')
f1_macro = f1_score(true_labels, pred_labels, average='macro')
f1_weighted = f1_score(true_labels, pred_labels, average='weighted')

print("Accuracy:", acc)
print("F1 Score:", f1_micro)
print("F1 Score:", f1_macro)
print("F1 Score:", f1_weighted)