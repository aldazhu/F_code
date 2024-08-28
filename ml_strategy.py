import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import random
import os
import numpy as np

from data_ml import MLDataset

from TCN_model import TemporalConvNet, MyTemporalConvNet


# class TemporalConvNet(nn.Module):
#     def __init__(self, input_size=5, output_size=1, num_channels=32, kernel_size=3):
#         super(TemporalConvNet, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.num_channels = num_channels
#         self.kernel_size = kernel_size

#         # Define the convolutional layers
#         self.conv1 = nn.Conv1d(input_size, num_channels, kernel_size, stride=2)
#         self.conv2 = nn.Conv1d(num_channels, num_channels*2, kernel_size, stride=2)
#         self.conv3 = nn.Conv1d(num_channels*2, num_channels, kernel_size, stride=2)

#         # Define the fully connected layer
#         self.fc = nn.Linear(num_channels, output_size)
#         self.leakyrelu = torch.nn.LeakyReLU()

#     def forward(self, x):
#         # Reshape the input to (batch_size, input_size, sequence_length)
#         # x = x.float()
#         x = x.permute(0, 2, 1)


#         # Apply the convolutional layers
#         x = self.leakyrelu(self.conv1(x))
#         x = self.leakyrelu(self.conv2(x))
#         x = self.leakyrelu(self.conv3(x))

#         # Apply global average pooling
#         x = F.avg_pool1d(x, x.shape[2])

#         # Flatten the output
#         x = x.view(x.shape[0], -1)

#         # Apply the fully connected layer
#         x = self.fc(x)
        
#         return x

def train_TCN():

    writer = SummaryWriter("runs/TCN")

    num_epochs = 100
    batch_size = 32
    pre_days = 30
    future_days = 9
    data_root = "data"
    use_catch = True
    train_ratio = 0.8
    learning_rate = 0.001
    num_channels = [32,32,32,32,32]
    kernel_size = 2
    dropout = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
       
    csv_files = [os.path.join(data_root, item) for item in os.listdir(data_root)]
    train_files = random.sample(csv_files, int(len(csv_files) * train_ratio))
    test_files = [file for file in csv_files if file not in train_files]

    train_dataset = MLDataset(train_files, pre_days, future_days, use_catch)
    test_dataset = MLDataset(test_files, pre_days, future_days, use_catch)

    # get data shape
    input_data, target = train_dataset[0]
    input_shape = input_data.shape
    target_shape = target.shape
    print(input_shape)
    print(target_shape)
    # tcn = TemporalConvNet(input_size=input_shape[1], output_size=target_shape[0], num_channels=32, kernel_size=3)
    feature_num = input_shape[1]
    # tcn = TemporalConvNet(feature_num, num_channels, kernel_size, dropout)
    tcn = MyTemporalConvNet(feature_num, future_days, num_channels, kernel_size, dropout)
    print(f"device: {device}")
    print(tcn)
    print(f"parameters: {sum(p.numel() for p in tcn.parameters())}")
    tcn.to(device)

    # Create data loaders

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the loss function and optimizer

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(tcn.parameters(), lr=learning_rate)

    # Train the TCN
    best_loss = float("inf")
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
                writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
        # Evaluate the model on the test
        tcn.eval()
        with torch.no_grad():
            total_loss = 0
            total_ic = 0
            total_ir = 0
            for x, y in test_loader:
                x = x.float().to(device)
                y = y.float().to(device)
                
                output = tcn(x)
                # calculate the IC and IR
                ic = np.corrcoef(output.cpu().numpy().flatten(), y.cpu().numpy().flatten())[0, 1]
                ir = 0
                total_ic += ic
                total_ir += ir
                loss = loss_function(output, y)
                total_loss += loss.item()
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(tcn.state_dict(), f"model/TCN/tcn_model_{epoch}.pth")
            
            writer.add_scalar("Loss/test", total_loss / len(test_loader), epoch)
            writer.add_scalar("IC", total_ic / len(test_loader), epoch)
            writer.add_scalar("IR", total_ir / len(test_loader), epoch)
            print(f"Epoch {epoch}, Test Loss: {total_loss / len(test_loader)}, IC: {total_ic / len(test_loader)}, IR: {total_ir / len(test_loader)}")
            print(f"Epoch {epoch}, Test Loss: {total_loss / len(test_loader)}")


if __name__ == "__main__":
    train_TCN()




