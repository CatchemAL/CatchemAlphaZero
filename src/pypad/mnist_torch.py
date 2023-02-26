import torch
import torch.nn as nn
import torch.optim as optim

from .mnist_loader import load_data
from .mnist_pytorch_loader import load_torch_data

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 125)
        self.fc2 = nn.Linear(125, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input image
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the training loop
def train(model, optimizer, cost_function, train_loader):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = cost_function(outputs, labels)
        loss.backward()
        optimizer.step()

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
    train_loader, test_loader = load_torch_data(batch_size)

    # Create the model, loss function, and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = MLP().to(device)
    print(model)

    cost_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha)

    # Train the model
    for epoch in range(num_epochs):
        train(model, optimizer, cost_function, train_loader)
        num_correct = test(model, test_loader)
        accuracy = num_correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1}: {accuracy*100:.2f}% accuracy")


