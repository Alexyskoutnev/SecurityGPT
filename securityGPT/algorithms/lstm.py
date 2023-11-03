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
        self.fc = nn.Linear(hidden_size, output_size)
        self.lstm_layer = nn.ModuleList([LSTM_Block(input_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x):
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

class Dataset(object):
    def __init__(self, seq_length = 100, batch_size=32):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data = torch.unsqueeze(torch.FloatTensor(np.array([generate_synthetic_data(seq_length) for _ in range(batch_size)])), dim=-1)
        self.target = torch.FloatTensor(np.roll(self.data, 1, axis=1))

    def load(self, ratio=0.2):
        end_idx = int((1 - ratio) * self.data.shape[1])
        train_data, train_labels = self.data[:,:end_idx], self.target[:,:end_idx]
        test_data, test_labels = self.data[:,end_idx:], self.target[:,end_idx:]
        return train_data, test_data, train_labels, test_labels

def train(model, dataset, epochs=100, lr=0.01):
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

def _test(samples):

    # Create a time array from 0 to 2*pi
    t = np.linspace(0, 2 * np.pi, samples)

    # Generate a sine wave that oscillates between -1 and 1
    oscillating_data = np.sin(t)
    # Scale the sine wave to oscillate between -1 and 1
    oscillating_data = (2 * oscillating_data - 1) 

    return oscillating_data


def plot(model, dataset):
    train_data, test_data, train_target, test_target = dataset.load()
    test = _test(200)
    desired_shape = test_data.shape
    _test_t = torch.from_numpy(np.reshape(test, desired_shape))
    _test_t = _test_t.to(torch.float32)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient tracking for inference
        pred_data = model(test_data)
        pred_data1 = model(_test_t)
    train_data = np.squeeze(train_data.numpy())
    pred_data = np.squeeze(pred_data.numpy())
    pred_data1 = np.squeeze(pred_data1.numpy())
    plt.plot(train_data, label='Input Data', linestyle='--')
    plt.plot(pred_data, label='Predicted Data', linestyle='-.')
    plt.plot(pred_data1, label="Outta Distribution")
    plt.legend()
    plt.show()
        
if __name__ == "__main__":
    input_size = 1
    hidden_size = 64
    batch_size = 1
    num_layers = 1
    output_size = 1
    epochs = 100
    # input = torch.randn(batch_size, 2, input_size)
     # 1. Create a synthetic dataset
    dataset = Dataset(seq_length=1000, batch_size=batch_size)
    lstm = LSTM(input_size, hidden_size, num_layers, output_size)
    plot(lstm, dataset)
    train(lstm, dataset, epochs=epochs)
    plot(lstm, dataset)