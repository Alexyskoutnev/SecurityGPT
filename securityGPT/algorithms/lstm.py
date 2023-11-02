import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Generate a synthetic dataset
def generate_synthetic_data(sequence_length):
    data = []
    for i in range(sequence_length):
        data.append(np.sin(0.1 * i) + np.random.normal(0, 0.1))
    return data

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
    
    def forward(self, input_seq : torch.tensor, prev_hidden : torch.tensor, prev_cell : torch.tensor):
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
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = LSTM_Block(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape
        self.lstm.seq_length = seq_length
        x = x.view(batch_size, seq_length, input_dim)
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h0 = torch.zeros(batch_size, self.hidden_size)
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(batch_size, self.hidden_size)
        hid = (h0, c0)
        out, hidden = self.lstm(x, *hid)
        out = self.fc(out)

        return out

class Dataset(object):
    def __init__(self, seq_length = 100):
        self.data = torch.FloatTensor(np.expand_dims(generate_synthetic_data(seq_length), axis=1))
        self.target = torch.FloatTensor(np.roll(self.data, 1))

    def load(self):
        return self.data, self.target

def train(model, dataset, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_data, train_train = dataset
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:   
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def plot(data, target):
    plt.plot(data, label='Input Data', linestyle='--')
    plt.plot(target, label='Target Data')
    plt.legend()
    plt.show()
        
if __name__ == "__main__":
    input_size = 32
    hidden_size = 64
    batch_size = 5
    num_layers = 1
    output_size = 1
    dataset = Dataset(100)
    data, target = dataset.load()
    # plot(data, target)
    input = torch.randn(batch_size, 2, input_size)
    lstm = LSTM(input_size, hidden_size, num_layers, output_size)
    out = lstm(input)
    breakpoint()