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

#### Utilities #### 
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Function to format time
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_stats(stats):
    df  = pd.DataFrame(data=stats)
    # Use the 'epoch' as the row index.
    df = df.set_index('epoch')
    return df    

# SET DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Reading all dataset
Scicite_data_getter = CreateDataset(filepath=FILEPATH)

# Getting all X and Y
train_label = Scicite_data_getter._label_to_id("train")
train_sentence = Scicite_data_getter._get_sentence_data("train")

valid_label = Scicite_data_getter._label_to_id("dev")
valid_sentence = Scicite_data_getter._get_sentence_data("dev")

# Tokenize
tokenize_utils = CreateTokenize(max_length=max_token_length, special_tokens=special_tokens)
train_input_ids, train_attention_masks, train_labels = tokenize_utils.xlnettokenize(
    all_sentences=train_sentence,labels=train_label)

valid_input_ids, valid_attention_masks, valid_labels = tokenize_utils.xlnettokenize(
    all_sentences=valid_sentence,labels=valid_label)

# Create dataloaders
dataloader_utils = CreateDataloaders(batch_num=batch_num)

train_dataloader = dataloader_utils.get_train_loader(
    input_ids=train_input_ids, attention_masks=train_attention_masks, labels=train_labels)
valid_dataloader = dataloader_utils.get_eval_dataloader(
    input_ids=valid_input_ids, attention_masks=valid_attention_masks, labels=valid_labels)

model = XLNetForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = len(Scicite_data_getter.classes), # The number of output labels
    output_attentions = False,
    output_hidden_states = False,
)

# Send to device
model.to(device)

optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = eps
                )

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = num_warmup_steps,
                                            num_training_steps = total_steps)


training_stats = []

# Measure the total training time for the whole run.
start_time = time.time()

# Training 
for epoch_i in range(0, epochs):

    print("")
    print(f'======== Epoch {epoch_i + 1} / {epochs} ========')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            print(f'Batch {step}  of  {len(train_dataloader)}.   Elapsed: {elapsed}.')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs.loss
        logits = outputs.logits

        total_train_loss += loss.item()

        loss.backward()
        
        # Clip the gradient to prevent expanding
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Total training of an epoch: {:}".format(training_time))

    print("Validation Results")

    t0 = time.time()

    # Put model into eval mode
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in valid_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Ensure no grad
        with torch.no_grad():
            outputs = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)

            loss = outputs.loss
            logits = outputs.logits

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(valid_dataloader)
    print(f"  Accuracy: {avg_val_accuracy}")

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(valid_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-start_time)))      

total_stats = get_stats(training_stats)

print(f" =========== Training Statistics ===============")
print(total_stats)

if save_flag:
    assert output_dir != None, "no output dir"

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save model to be loaded later 
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)