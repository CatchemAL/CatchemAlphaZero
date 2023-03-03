import torch
import torch.nn as nn
import torch.optim as optim

from .mnist_loader import load_data
from .mnist_pytorch_loader import load_torch_data

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.network_layers = nn.Sequential(
            nn.Linear(784, 125),
            nn.ReLU(),
            nn.Linear(125, 10),
        )

    def forward(self, x):
        x = self.network_layers(x)
        return x

# Define the training loop
def train(model, optimizer, cost_function, train_loader):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = cost_function(predictions, labels)
        loss.backward() # calculate gradients
        optimizer.step() # update weights

# Define the testing loop
def test(model, test_loader):
    model.eval()
    num_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            num_correct += torch.sum(predictions == labels).item()
    return num_correct

def run_torch():

    # Define the hyperparameters
    learning_rate = 0.22
    alpha = 0.0005
    num_epochs = 10
    batch_size = 32

    # Create the data loaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = load_torch_data(batch_size, device)

    # Create the model, loss function, and optimizer
    print(f"Using {device} device")
    model = MLP().to(device)
    print(model)

    cost_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha)

    for epoch in range(num_epochs):
        # Train the model
        train(model, optimizer, cost_function, train_loader)
    
        # Test after each epoch
        num_correct = test(model, test_loader)
        accuracy = num_correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1}: {accuracy*100:.2f}% accuracy")


