import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

dataset_path = os.path.join("./data")
dataset_output = os.path.join("./data/graphs/")

from securityGPT.algorithms.mlp import MLP
from securityGPT.dataset import Loader
from securityGPT.utils.utils import save_model

def plot_loss(data : dict):
    path = os.path.join(dataset_output)
    for key in data.keys():
        epochs = np.arange(len(data[key]))
        plt.plot(epochs, data[key], label='Line Graph')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(path + str(key) + "_mlp_" + ".png")
        plt.close()

def plot_acc(data : dict):
    path = os.path.join(dataset_output)
    for key in data.keys():
        epochs = np.arange(len(data[key]))
        plt.plot(epochs, data[key], label='Line Graph')
        plt.xlabel('Epochs')
        plt.ylabel('Accurancy (%)')
        plt.savefig(path + str(key) + "_mlp_" + ".png")
        plt.close()

def main():
    size = 500
    ratio = 0.1
    EVAL_STEP = 2
    batch_size = 64
    dataloader = Loader(dataset_path, size=size, torch=True,
                        batch_size=batch_size, doc2vec=True,
                        padding_bool=True)
    X_train, y_train, X_test, y_test, data = dataloader.load(bootstrap=True, ratio=ratio)
    input_dim = X_train.shape[1]
    mlp = MLP(input_dim, output_size=1)

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)

    all_loss = {'train_loss':[], 'testing_loss':[]}
    all_acc = {'train_acc':[], 'testing_acc':[]}
    all_f1 = {"train_f1" : [], "testing_f1" : []}
    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        for _X_train, _y_train in data():
            optimizer.zero_grad()
            outputs = torch.FloatTensor(torch.squeeze(torch.squeeze(mlp(_X_train), dim=1)))
            loss = criterion(outputs, _y_train.float())
            loss.backward()
            optimizer.step()
        if epoch % EVAL_STEP == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            with torch.no_grad():
                test_outputs = torch.squeeze(torch.sigmoid(mlp(X_test)), dim=1)
                predicted_labels = (test_outputs >= 0.5).int()  # Convert probabilities to binary labels (0 or 1)
            # Calculate accuracy and F1 score
            accuracy = accuracy_score(y_test, predicted_labels.numpy())
            f1 = f1_score(y_test, predicted_labels.numpy())
            print(f'Test Accuracy: {accuracy:.2f}')
            print(f'F1 Score: {f1:.2f}')
            #============= logging ============= 
            all_loss['train_loss'].append(loss.item())
            all_acc['testing_acc'].append(accuracy)
            #============= logging ============= 
    try: 
        plot_loss(all_loss)
        plot_acc(all_acc)
        save_model(mlp.net, type="mlp")
    except:
        print("Error in plotting and saving")        

if __name__ == "__main__":
    main()