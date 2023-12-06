import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from securityGPT.dataset import Loader

dataset_path = os.path.join("../../data")
dataset_folder = os.path.join("../../models/mlp")

class MLP(nn.Module):
    
    def __init__(self, input_size, output_size, dropout_rate=0.2) -> None:
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
            # nn.Softmax(dim=1),
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

if __name__ == "__main__":
    size = 0
    ratio = 0.3
    EVAL_STEP = 10
    batch_size = 64
    dataloader = Loader(dataset_path, size=size, torch=True,
                        word_embedding=False, batch_size=batch_size,
                        padding_bool=False)
    X_train, y_train, X_test, y_test, data = dataloader.load(bootstrap=True, ratio=ratio)
    input_dim = X_train.shape[2]
    mlp = MLP(input_dim, output_size=1)

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.0001)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        for _X_train, _y_train in data():
            optimizer.zero_grad()
            outputs = torch.FloatTensor(torch.squeeze(torch.squeeze(mlp(_X_train), dim=1)))
            loss = criterion(outputs, _y_train.float())
            loss.backward()
            optimizer.step()
            breakpoint()
        if epoch % EVAL_STEP == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            with torch.no_grad():
                test_outputs = torch.squeeze(torch.sigmoid(mlp(X_test)), dim=1)
                predicted_labels = torch.squeeze((test_outputs >= 0.5).int(), dim=1)  # Convert probabilities to binary labels (0 or 1)
            # Calculate accuracy and F1 score
            accuracy = accuracy_score(y_test, predicted_labels.numpy())
            f1 = f1_score(y_test, predicted_labels.numpy())
            print(f'Test Accuracy: {accuracy:.2f}')
            print(f'F1 Score: {f1:.2f}')





