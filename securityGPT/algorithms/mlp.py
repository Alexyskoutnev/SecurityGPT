import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self, input_size : int, output_size : int, dropout_rate : float  = 0.2) -> None:
        """
        Multi-Layer Perceptron (MLP) model with dropout.

        Args:
            input_size (int): The number of features in the input data.
            output_size (int): The number of output classes.
            dropout_rate (float, optional): Dropout rate to apply between layers. Defaults to 0.2.

        Note:
            The architecture consists of three hidden layers with ReLU activation functions
            and dropout regularization between each layer.

        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.

        """
        return self.net(x)