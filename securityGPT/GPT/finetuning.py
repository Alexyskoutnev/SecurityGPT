import io
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from securityGPT.dataset import DataTorch
from typing import Union, Optional

dataset_path = os.path.join("./data")
dataset_output = os.path.join("./data/graphs/")
SAVE_DIR = "./models/"

def gpt_save(model, type="gpt"):
    time = str(datetime.now())
    path = os.path.join(SAVE_DIR, type, time + "_" +  type + ".pth")
    torch.save(model.state_dict(), path)

def plot_loss(data : dict):
    path = os.path.join(dataset_output)
    for key in data.keys():
        epochs = np.arange(len(data[key]))
        plt.plot(epochs, data[key], label='Line Graph')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(path + str(key) + "_gpt_" + ".png")
        plt.close()

def plot_acc(data : dict, plot_type="testing_acc"):
    path = os.path.join(dataset_output)
    epochs = np.arange(len(data[plot_type]))
    plt.plot(epochs, data[plot_type], label='Line Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Accurancy (%)')
    plt.savefig(path + str(plot_type) + "_gpt_" + ".png")
    plt.close()

def train(model, dataloader : DataLoader, optimizer_ : torch.optim, scheduler_, device_):
  r"""
  Train pytorch model on a single pass through the data loader.

  It will use the global variable `model` which is the transformer model 
  loaded on `_device` that we want to train on.

  This function is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

  Arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.

      optimizer_ (:obj:`transformers.optimization.AdamW`):
          Optimizer used for training.

      scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
          PyTorch scheduler.

      device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.

  Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss].
  """
  predictions_labels = []
  true_labels = []
  total_loss = 0
  model.train()
  for batch in dataloader:
    true_labels += batch['labels'].numpy().flatten().tolist()
    batch = {k:v.type(torch.long) for k,v in batch.items()}
    model.zero_grad()
    outputs = model(**batch)
    loss, logits = outputs[:2]
    total_loss += loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer_.step()
    scheduler_.step()
    logits = logits.detach().cpu().numpy()
    predictions_labels += logits.argmax(axis=-1).flatten().tolist()
  avg_epoch_loss = total_loss / len(dataloader)
  return true_labels, predictions_labels, avg_epoch_loss

class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer : object, labels_encoder : object, max_sequence_len : Optional[Union[int, None]] = None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder

    def __call__(self, sequences : dict) -> dict:
        r"""
        This function allowes the class object to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        labels = [self.labels_encoder[label] for label in labels]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        inputs.update({'labels':torch.tensor(labels)})
        return inputs

def validation(model, dataloader : DataLoader, device : dict):
    predictions_labels = []
    true_labels = []
    total_loss = 0.0
    model.eval()
    for batch in (dataloader):
        true_labels += batch['labels'].numpy().flatten().tolist()
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content
    avg_epoch_loss = total_loss / len(dataloader)
    return true_labels, predictions_labels, avg_epoch_loss