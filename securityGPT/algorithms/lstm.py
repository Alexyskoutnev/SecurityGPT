import os
import sys
from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from securityGPT.dataset import Loader
from sklearn.metrics import accuracy_score, f1_score

dataset_path = os.path.join("../../data")
dataset_folder = os.path.join("../../models/svm")

# Generate a synthetic dataset
def generate_synthetic_data(sequence_length):
    data = []
    for i in range(sequence_length):
        data.append(np.sin(0.1 * i) + np.random.normal(0, 0.1))
    return data

def normalized(model):
    for layer in model._all_weights:
        for param in layer:
            if 'weight' in param:
                init.xavier_normal_(getattr(model, param))

class Torch_LSTM_Classifer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(Torch_LSTM_Classifer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        normalized(self.lstm)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = torch.sigmoid(out)
        return out

class LSTM_Block(nn.Module):
    def __init__(self, input_size : int, hidden_size : int) -> None:
        super(LSTM_Block, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = 1
        #Input Gate
        self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        #Forget Gate
        self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        #Cell State
        self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))

        #Output Gate
        self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()
    
    def reset_parameters(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
    
    def forward(self, input_seq : torch.Tensor, prev_hidden : torch.Tensor, prev_cell : torch.Tensor):
        hidden_states = []
        cell_states = []
        for t in range(self.seq_length):
            input = input_seq[:, t, :]
            # Input Gate
            i_t = torch.sigmoid(torch.mm(input, self.W_ii) + torch.mm(prev_hidden, self.W_hi) + self.b_i)
            # Forget Gate
            f_t = torch.sigmoid(torch.mm(input, self.W_if) + torch.mm(prev_hidden, self.W_hf) + self.b_f)
            # Cell state update
            g_t = torch.tanh(torch.mm(input, self.W_ig) + torch.mm(prev_hidden, self.W_hg) + self.b_g)
            new_cell = f_t * prev_cell + i_t * g_t
            # Output Gate
            o_t = torch.sigmoid(torch.mm(input, self.W_io) + torch.mm(prev_hidden, self.W_ho) + self.b_o)
            new_hidden = o_t + torch.tanh(new_cell)
            hidden_states.append(new_hidden)
            cell_states.append(new_cell)
        hidden_states = torch.stack(hidden_states, dim=1)
        cell_states = torch.stack(cell_states, dim=1)
        return hidden_states, cell_states

class LSTM(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, num_layers : int, output_size : int) -> None:
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.fc = nn.Linear(hidden_size, output_size)
        self.lstm_layer = nn.ModuleList([LSTM_Block(input_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, input_dim = x.shape
        for layer in self.lstm_layer:
            layer.seq_length = seq_length
        x = x.view(batch_size, seq_length, input_dim)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        output = x
        for i, lstm_i in enumerate(self.lstm_layer):
            hid = (h0[i], c0[i])
            x, _ = lstm_i(output, *hid)
            if i < len(self.lstm_layer) - 1:
                output = x.view(batch_size, seq_length, -1)
            else:
                output = x
        out = self.fc(output)
        return out


class LSTMClassifer(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, num_layers : int, output_size : int) -> None:
        super(LSTMClassifer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.fc = nn.Linear(hidden_size, output_size)
        self.lstm_layer = nn.ModuleList([LSTM_Block(input_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, input_dim = x.shape
        for layer in self.lstm_layer:
            layer.seq_length = seq_length
        x = x.view(batch_size, seq_length, input_dim)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        output = x
        for i, lstm_i in enumerate(self.lstm_layer):
            hid = (h0[i], c0[i])
            # breakpoint()
            x, _ = lstm_i(output, *hid)
            if i < len(self.lstm_layer) - 1:
                output = x.view(batch_size, seq_length, -1)
            else:
                output = x
        out = self.fc(output[:, -1, :])
        out = torch.sigmoid(out)
        return out

class Dataset(object):
    def __init__(self, seq_length = 100, batch_size=32) -> None:
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data = torch.unsqueeze(torch.FloatTensor(np.array([generate_synthetic_data(seq_length) for _ in range(batch_size)])), dim=-1)
        self.target = torch.FloatTensor(np.roll(self.data, 1, axis=1))

    def load(self, ratio=0.2) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        end_idx = int((1 - ratio) * self.data.shape[1])
        train_data, train_labels = self.data[:,:end_idx], self.target[:,:end_idx]
        test_data, test_labels = self.data[:,end_idx:], self.target[:,end_idx:]
        return train_data, test_data, train_labels, test_labels

def train(model : LSTM, dataset : object, epochs : int = 100, lr : float = 0.01) -> None:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_data, _, train_target, _ = dataset.load()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_target)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:   
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def plot(model : LSTM, dataset : object) -> None:
    train_data, test_data, train_target, test_target = dataset.load()
    model.eval() 
    with torch.no_grad():
        pred_data = model(test_data)
    train_data = np.squeeze(train_data.numpy())
    pred_data = np.squeeze(pred_data.numpy())
    plt.plot(train_data, label='Input Data', linestyle='--')
    plt.plot(pred_data, label='Predicted Data', linestyle='-.')
    plt.legend()
    plt.show()
        
if __name__ == "__main__":
    size = 1000
    ratio = 0.3
    EVAL_STEP = 10
    batch_size = 32
    dataloader = Loader(dataset_path, size=size, torch=True, word_embedding=True, batch_size=batch_size)
    X_train, y_train, X_test, y_test, data = dataloader.load(bootstrap=True, ratio=ratio)
    input_dim = X_train.shape[2]
    hidden_dim = 64
    output_dim = 2  # Adjust based on the number of classes (e.g., binary classification)
    num_layers = 2
    lstm = Torch_LSTM_Classifer(input_dim, hidden_dim, num_layers, output_dim)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm.parameters(), lr=0.001)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        for X_train, y_train in data():
            optimizer.zero_grad()
            outputs = lstm(X_train)
            max_prob_indices = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        if epoch % EVAL_STEP == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
             # Make predictions on the test set
            with torch.no_grad():
                test_outputs = lstm(X_test)
                predicted_labels = torch.argmax(test_outputs, 1)
            # Calculate accuracy
            accuracy = accuracy_score(y_test, predicted_labels.numpy())
            print(f'Test Accuracy: {accuracy:.2f}')
            f1 = f1_score(y_test, predicted_labels.numpy())
            print(f'F1 Score: {f1:.2f}')


        