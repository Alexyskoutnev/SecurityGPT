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
from securityGPT.GPT.finetuning import plot_loss, plot_acc, train, validation, Gpt2ClassificationCollator, gpt_save
from typing import Union, Optional

def main():
    #============== Config =============
    set_seed(np.random.randint(0, 10000))
    epochs = 1
    batch_size = 32
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
    dataset_path = os.path.join("./data")
    size = 1000
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
        # all_acc['train_acc'].append(train_loss)
        # all_loss['testing_loss'].append(testing_loss)
        all_acc['testing_acc'].append(testing_acc)
        all_f1['train_f1'].append(f1_training)
        all_f1['testing_f1'].append(f1_testing)
        #================== Logging ================== 
    #=============== Training Loop =========  
    print("GPT has been finetuned")
    try: 
        plot_loss(all_loss)
        plot_acc(all_acc)
        gpt_save(model)
    except:
        print("Failed to plot and save model")

if __name__ == "__main__":
    main()