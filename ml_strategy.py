import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import random
import os
import numpy as np
import matplotlib.pyplot as plt

from data_ml import MLDataset

from ml_model import TemporalConvNet, MyTemporalConvNet, MLP


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

def train(model, train_loader, test_loader, loss_function, optimizer, num_epochs, writer, device, save_dir):
    best_loss = float("inf")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(num_epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            # Forward pass
            output = model(x)

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
        model.eval()
        total_count = 0
        success_count = 0
        with torch.no_grad():
            total_loss = 0
            total_ic = 0
            total_ir = 0
            for x, y in test_loader:
                x = x.float().to(device)
                y = y.float().to(device)
                
                output = model(x)
                # calculate the IC and IR
                ic = np.corrcoef(output.cpu().numpy().flatten(), y.cpu().numpy().flatten())[0, 1]
                ir = 0
                total_ic += ic
                total_ir += ir
                # print(f"output[-1]: {output[-1]}")
                # print(f"output[:,-1]: {output[:,-1]}")
                loss = loss_function(output, y)
                total_loss += loss.item()
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), f"{save_dir}/model_best.pth")

            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"{save_dir}/model_{epoch}.pth")

            total_count += y.shape[0]
            if y[-1] * output[-1] > 0:
                success_count += 1
            print(f"success_count: {success_count} / {total_count}")
            
            writer.add_scalar("Loss/test", total_loss / len(test_loader), epoch)
            writer.add_scalar("IC", total_ic / len(test_loader), epoch)
            writer.add_scalar("IR", total_ir / len(test_loader), epoch)
            print(f"Epoch {epoch}, Test Loss: {total_loss / len(test_loader)}, IC: {total_ic / len(test_loader)}, IR: {total_ir / len(test_loader)}")
            print(f"Epoch {epoch}, Test Loss: {total_loss / len(test_loader)}")


def test(model, test_loader,  device):
    model.to(device)
    model.eval()
    label = []
    predict = []
    total_count = 0
    success_count = 0
    with torch.no_grad():
        total_loss = 0
        total_ic = 0
        total_ir = 0
        for x, y in test_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            
            output = model(x)
            # calculate the IC and IR
            ic = np.corrcoef(output.cpu().numpy().flatten(), y.cpu().numpy().flatten())[0, 1]
            ir = 0
            total_ic += ic
            total_ir += ir
            label.extend(y.cpu().numpy().flatten())
            predict.extend(output.cpu().numpy().flatten())

            total_count += y.shape[0]
            if y[-1] * output[-1] > 0:
                success_count += 1

            print(f"output : {output}")
            print(f"y : {y}")
            print(f" {success_count} / {total_count}")
            input("Press Enter to continue...")
            
        print(f"IC: {total_ic / len(test_loader)}, IR: {total_ir / len(test_loader)}")
    plt.scatter(label, predict)
    plt.show()


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

    save_dir = "model/TCN"
    train(tcn, train_loader, test_loader, loss_function, optimizer, num_epochs, writer, device, save_dir)


def train_MLP():
    
    save_dir = "model/MLP"
    writer = SummaryWriter(save_dir)
    num_epochs = 100
    batch_size = 32
    pre_days = 10
    future_days = 5
    data_root = "data"
    use_catch = False
    use_signal_future_day = True # use the last day as the target
    train_ratio = 0.8
    learning_rate = 0.001
    hidden_dim = 512
    output_dim = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
       
    csv_files = [os.path.join(data_root, item) for item in os.listdir(data_root)]
    train_files = random.sample(csv_files, int(len(csv_files) * train_ratio))
    test_files = [file for file in csv_files if file not in train_files]
    with open(f"{save_dir}/train_files.txt", "w") as f:
        for file in train_files:
            f.write(file + "\n")
    with open(f"{save_dir}/test_files.txt", "w") as f:
        for file in test_files:
            f.write(file + "\n")

    train_dataset = MLDataset(train_files, pre_days, future_days, use_catch, use_signal_future_day, npy_save_prefix=f"{save_dir}/train")
    test_dataset = MLDataset(test_files, pre_days, future_days, use_catch, use_signal_future_day, npy_save_prefix=f"{save_dir}/test")

    # get data shape
    input_data, target = train_dataset[0]
    print(f"input_data shape: {input_data.shape}")
    print(f"target shape: {target.shape}")
    print(f"target: {target}")
    input_shape = input_data.shape
    target_shape = target.shape
    output_dim = target_shape[0] if len(target_shape) > 1 else 1
    mlp = MLP(input_shape[1], pre_days, hidden_dim, output_dim)
    print(f"device: {device}")
    print(mlp)
    print(f"parameters: {sum(p.numel() for p in mlp.parameters())}")
    mlp.to(device)

    # Create data loaders

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the loss function and optimizer

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    train(mlp, train_loader, test_loader, loss_function, optimizer, num_epochs, writer, device, save_dir)

def test_MLP():
    batch_size = 1
    pre_days = 10
    future_days = 5
    use_catch = True
    hidden_dim = 512

    save_dir = "model/MLP"  
    model_path = f"{save_dir}/model_best.pth"
    test_csv_file_dict = f"{save_dir}/train_files.txt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    test_files = []
    with open(test_csv_file_dict, "r") as f:
        for line in f:
            test_files.append(line.strip())
    
    test_dataset = MLDataset(test_files, pre_days, future_days, use_catch, use_signal_future_day=True, npy_save_prefix=f"{save_dir}/test")
    input_data, target = test_dataset[0]
    input_shape = input_data.shape
    target_shape = target.shape
    print(input_shape)
    print(target_shape)
    output_dim = target_shape[0] if len(target_shape) > 1 else 1
    mlp = MLP(input_shape[1], pre_days, hidden_dim, output_dim)
    mlp.load_state_dict(torch.load(model_path))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test(mlp, test_loader,  device)

if __name__ == "__main__":
    # train_TCN()
    # train_MLP()
    test_MLP()




