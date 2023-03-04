import torch
import torch.nn as nn
import torch.optim as optim

from .mnist_loader import load_data


def load_torch_data(batch_size: int = 32, device: str = "cpu"):
    # load data
    training_data, validation_data, test_data = load_data()
    x_train, y_train = training_data[0][:, :], training_data[1][:]
    x_test, y_test = test_data[0], test_data[1]

    # Convert data to PyTorch tensors
    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    # Create the data loaders
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
