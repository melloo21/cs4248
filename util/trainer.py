import os
import pandas as pd
from tqdm import tqdm,trange
from sklearn.metrics import classification_report
import torch
from torch.optim import Adam
import torch.nn.functional as F

class ModelTrainer:
  def __init__(
      self,
      model,
      savepath:str,
      batch_num:int,
      input_length:int,
      max_grad_norm:int=1,
      full_fine_tuning:bool=True
  ):
    self.model = model
    self.savepath = savepath
    self.batch_num = batch_num
    self.input_length = input_length
    self.max_grad_norm = max_grad_norm
    self.full_fine_tuning = full_fine_tuning

  def _get_optimizer(self):
    if self.full_fine_tuning:
      # Fine tune model all layer parameters
      param_optimizer = list(self.model.named_parameters())
      no_decay = ['bias', 'gamma', 'beta']
      return [
          {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.01},
          {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.0}
      ]
    else:
      # Only fine tune classifier parameters -- last layer
      param_optimizer = list(self.model.classifier.named_parameters())
      return [{"params": [p for n, p in param_optimizer]}]

  # Set acc funtion
  def accuracy(self, out, labels):
    out = out.detach().cpu().numpy()
    labels = labels.to('cpu').numpy()
    # accuracy(logits, label_ids)
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

  @torch.no_grad()
  def evaluate(self, model, data_loader):
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    y_true = []
    y_predict = []

    model.eval()
    for step, batch in enumerate(data_loader):
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_segs,b_labels = batch

      with torch.no_grad():
          outputs = model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
          tmp_eval_loss, logits = outputs[:2]

      # Get textclassification predict result
      # logits = logits.detach().cpu().numpy()
      # label_ids = b_labels.to('cpu').numpy()
      tmp_eval_accuracy = self.accuracy(logits, b_labels)

      eval_loss += tmp_eval_loss.mean().item()
      eval_accuracy += tmp_eval_accuracy

      nb_eval_steps += 1
      eval_loss = eval_loss / nb_eval_steps
      eval_accuracy = eval_accuracy / len(b_labels)

      return {"valid_loss" : eval_loss,  "valid_accuracy" : eval_accuracy}

  def train_model(
      self,
      train_loader,
      valid_loader,
      learning_rate,
      epoch_num
      ):

    # Meta data
    best_valid = None
    history = []

    # Initalise optimiser
    optimizer_grouped_parameters = self._get_optimizer()
    optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

    self.model.to(device)

    # Fine tuning the model
    self.step_num = int( math.ceil(self.input_length  / self.batch_num) / 1) * epoch_num
    print(f"""
    ***** Running training ***** \n
    Num Examples {self.input_length } \n
    Batch Size {self.batch_num} and Num steps {self.step_num}
    """)

    for _ in trange(epoch_num,desc="Epoch"):
      # Set model to train
      self.model.train()
      tr_loss ,tr_acc = 0, 0
      nb_tr_examples, nb_tr_steps = 0, 0

      for step, batch in enumerate(train_loader):
          # add batch to gpu
          batch = tuple(t.to(device) for t in batch)
          b_input_ids, b_input_mask, b_segs,b_labels = batch

          # forward pass
          outputs = self.model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
          loss, logits = outputs[:2]

          if n_gpu>1:
              # When multi gpu, average it
              loss = loss.mean()

          # backward pass
          loss.backward()

          # track train loss
          tr_loss += loss.item()
          # track train accuarcy
          tr_acc += self.accuracy(logits, b_labels)

          nb_tr_examples += b_input_ids.size(0)
          nb_tr_steps += 1

          if nb_tr_steps % 10 == 0:
            print(f"current train loss {tr_loss/nb_tr_steps} and acc {tr_acc/nb_tr_steps}")

          # gradient clipping
          torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)

          # update parameters
          optimizer.step()
          optimizer.zero_grad()

      # Every EPOCH should evaluate Valid data loader
      results = self.evaluate(self.model, valid_loader)
      results["train_loss"] = tr_loss/nb_tr_steps
      results["train_acc"]  = tr_acc/nb_tr_steps
      # print train loss per epoch
      print(f" results : {results}")

      if(best_valid == None or best_valid<results['valid_accuracy']):
        best_valid=results['valid_accuracy']
        torch.save(self.model, f"{self.savepath}.pt")
        torch.save(self.model.state_dict(), f"{self.savepath}_state_dict.pt")
      # Saving results
      history.append(results)

    return history