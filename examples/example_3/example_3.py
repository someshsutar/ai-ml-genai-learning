import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# -----------------------------
# Step 1: Load Dataset (Inputs)
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# Step 2: Define Model (Features → Prediction)
# -----------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # weights & bias for first layer
        self.fc2 = nn.Linear(128, 64)     # weights & bias for hidden layer
        self.fc3 = nn.Linear(64, 10)      # weights & bias for output (10 digits)

    def forward(self, x):
        x = x.view(-1, 28*28)             # flatten image into vector (features)
        x = torch.relu(self.fc1(x))       # apply weights+bias, then activation
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)                   # final scores for 10 classes
        return x

model = SimpleNN()

# -----------------------------
# Step 3: Define Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Step 4: Train Model
# -----------------------------
for epoch in range(2):  # small number of epochs for demo
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/2], Loss: {loss.item():.4f}')

# -----------------------------
# Step 5: Evaluate Model
# -----------------------------
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')
