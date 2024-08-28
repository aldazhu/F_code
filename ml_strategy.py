import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import os

from data_ml import MLDataset


class TemporalConvNet(nn.Module):
    def __init__(self, input_size=5, output_size=1, num_channels=32, kernel_size=3):
        super(TemporalConvNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size

        # Define the convolutional layers
        self.conv1 = nn.Conv1d(input_size, num_channels, kernel_size)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size)
        self.conv3 = nn.Conv1d(num_channels, num_channels, kernel_size)

        # Define the fully connected layer
        self.fc = nn.Linear(num_channels, output_size)
        self.leakyrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        # Reshape the input to (batch_size, input_size, sequence_length)
        # x = x.float()
        x = x.permute(0, 2, 1)


        # Apply the convolutional layers
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))

        # Apply global average pooling
        x = F.avg_pool1d(x, x.shape[2])

        # Flatten the output
        x = x.view(x.shape[0], -1)

        # Apply the fully connected layer
        x = self.fc(x)
        
        return x

def train_TCN():

    num_epochs = 100
    batch_size = 32
    pre_days = 20
    future_days = 10
    data_root = "data"
    train_ratio = 0.8
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # Example usage
    # # Create a TCN with input size 5（OHLCV）, output size 1, 32 channels, and kernel size 3
    # tcn = TemporalConvNet(input_size=5, output_size=1, num_channels=32, kernel_size=3)

    # # Create some sample input data (batch_size, sequence_length, input_size)
    # input_data = torch.randn(16, 20, 5)

    # # Pass the input data through the TCN
    # output = tcn(input_data)

    # # Print the output shape
    # print(output.shape)  # Output shape: (16, 1)

    
    csv_files = [os.path.join(data_root, item) for item in os.listdir(data_root)]
    train_files = random.sample(csv_files, int(len(csv_files) * train_ratio))
    test_files = [file for file in csv_files if file not in train_files]

    train_dataset = MLDataset(train_files, pre_days, future_days)
    test_dataset = MLDataset(test_files, pre_days, future_days)

    # get data shape
    input_data, target = train_dataset[0]
    input_shape = input_data.shape
    target_shape = target.shape
    print(input_shape)
    print(target_shape)
    tcn = TemporalConvNet(input_size=input_shape[1], output_size=target_shape[0], num_channels=32, kernel_size=3)
    print(f"device: {device}")
    tcn.to(device)

    # Create data loaders

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the loss function and optimizer

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(tcn.parameters(), lr=learning_rate)

    # Train the TCN
    for epoch in range(num_epochs):
        tcn.train()
        for i, (x, y) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)


            # Forward pass
            output = tcn(x)

            # Compute the loss
            loss = loss_function(output, y)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Print the loss every 100 iterations
            if i % 100 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
        # Evaluate the model on the test
        tcn.eval()
        with torch.no_grad():
            total_loss = 0
            for x, y in test_loader:
                x = x.float().to(device)
                y = y.float().to(device)
                output = tcn(x)
                loss = loss_function(output, y)
                total_loss += loss.item()
            print(f"Epoch {epoch}, Test Loss: {total_loss / len(test_loader)}")


if __name__ == "__main__":
    train_TCN()




