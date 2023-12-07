import io
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
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

def plot_loss(data):
    for key in data.keys():
        epochs = np.arange(len(data[key]))
        plt.plot(epochs, data[key], label='Line Graph')
        plt.xlabel('Epochs Loss')
        plt.ylabel('Loss')
        plt.show()

def plot_acc(data):
    for key in data.keys():
        epochs = np.arange(len(data[key]))
        plt.plot(epochs, data[key], label='Line Graph')
        plt.xlabel('Epochs Loss')
        plt.ylabel('Accurancy')
        plt.show()

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
    # Update the learning rate.
    scheduler_.step()
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    # Convert these logits to list of predicted labels values.
    predictions_labels += logits.argmax(axis=-1).flatten().tolist()
  # Calculate the average loss over the training data.
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

if __name__ == "__main__":
    #============== Config =============
    set_seed(123)
    epochs = 10
    batch_size = 10
    max_length = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_or_path = 'gpt2'
    labels_ids = {'0': 0, '1': 1}
    n_labels = len(labels_ids)
    #============== Config ===============
    #============== Loading Model ========
    print('Loading configuraiton...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    print('Model loaded to `%s`'%device)
    #============== Setup =================
    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)
    dataset_path = os.path.join("../../data")
    size = 100
    #============== Loading Dataset ============== 
    train_dataset = DataTorch(dataset_path, size=size, use_tokenizer=None,
                              bootstrap=True, ratio=0.2, train_data=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=gpt2_classificaiton_collator)
    test_dataset = DataTorch(dataset_path, size=size, use_tokenizer=None,
                              bootstrap=True, ratio=0.2, test_data=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=gpt2_classificaiton_collator)
    #============== Loading Dataset ============== 
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))
    optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # default is 5e-5
                  eps = 1e-8 # default is 1e-8.
                  )
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
    all_loss = {'train_loss':[], 'testing_loss':[]}
    all_acc = {'train_acc':[], 'testing_acc':[]}
    all_f1 = {"train_f1" : [], "testing_f1" : []}
    #=============== Training Loop =========
    for epoch in range(epochs):
        # Training Eval
        train_labels, train_predict, train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)
        # Testing Eval
        testing_labels, testing_predict, testing_loss = validation(model, test_dataloader, device)
        testing_acc = accuracy_score(testing_labels, testing_predict)
        print("train_loss: %.5f - train_acc: %.5f | testing_loss: %.5f - testing_acc %.5f"%(train_loss, train_acc, testing_loss,  testing_acc))
        f1_testing = f1_score(testing_labels, testing_predict)
        f1_training = f1_score(train_labels, train_predict)
        print("training f1: %.5f  -  testing f1: %.5f" % (f1_training, f1_testing))
        #================== Logging ================== 
        all_loss['train_loss'].append(train_loss)
        all_acc['train_acc'].append(train_loss)
        all_loss['testing_loss'].append(testing_loss)
        all_acc['testing_acc'].append(testing_acc)
        all_f1['train_f1'].append(f1_training)
        all_f1['testing_f1'].append(f1_testing)
        #================== Logging ================== 
    #=============== Training Loop =========    
    print("GPT has been finetuned")
    plot_loss(all_loss)
    plot_acc(all_acc)