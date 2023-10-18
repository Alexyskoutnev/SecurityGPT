import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.Wx = nn.Parameter(torch.randn(hidden_size, input_size))
        self.Wh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wy = nn.Parameter(torch.randn(output_size, hidden_size))
        self.bh = nn.Parameter(torch.zeros(hidden_size, 1))
        self.by = nn.Parameter(torch.zeros(output_size, 1))

    def forward(self, x):
        T = x.size(0)
        hidden_states = []

        h_t = torch.zeros(self.hidden_size, 1)

        for t in range(T):
            breakpoint()
            x_t = x[t].view(-1, 1)
            h_t = torch.tanh(torch.mm(self.Wx, x_t) + torch.mm(self.Wh, h_t) + self.bh)
            y_t = torch.mm(self.Wy, h_t) + self.by
            hidden_states.append(h_t)   
        
        hidden_states = torch.stack(hidden_states, dim=1)
        return hidden_states, y_t

def train(model, X, y, epochs=1000, lr=0.1, loss_fn=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        hidden_state, preds = model(X)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

if __name__ == "__main__":
    input_size, hidden_size, output_size = 2, 4, 1
    lr, epochs = 0.01, 1000
    lost_fn = nn.MSELoss()
    rnn = RNN(input_size, hidden_size, output_size)
    test_x, test_y = torch.randn(size=(5,2,1)), torch.tensor([[1.0], [2.0], [1.0], [1.0]])
    train(model=rnn, X=test_x, y=test_y, loss_fn=lost_fn)
    print(rnn(test_x + torch.randn((5, 2, 1))))