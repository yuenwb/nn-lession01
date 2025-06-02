# Handwritten Digit Odd/Even Classification (MNIST, PyTorch, CPU)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Prepare Data
def even_odd_target(target):
    return target % 2

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True,
    transform=transform,
    target_transform=even_odd_target
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True,
    transform=transform,
    target_transform=even_odd_target
)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. Build Model
class OddEvenNet(nn.Module):
    def __init__(self):
        super(OddEvenNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)  # 1 output (odd or even)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # No sigmoid here

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OddEvenNet().to(device)

# 3. Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {running_loss/len(train_loader):.4f}  Train Acc: {train_acc:.4f}")

# 5. Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)
test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")

# 6. Visualize Predictions
import numpy as np

def show_images(images, labels, preds=None):
    plt.figure(figsize=(10,2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.axis('off')
        plt.imshow(images[i].squeeze(), cmap='gray')
        title = "Odd" if labels[i].item() == 1 else "Even"
        if preds is not None:
            pred_text = "Odd" if preds[i].item() == 1 else "Even"
            title += f"\n({pred_text})"
        plt.title(title, fontsize=8)
    plt.show()

# Show a batch with predictions
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
outputs = model(images)
preds = (torch.sigmoid(outputs) > 0.5).long()
show_images(images.cpu(), labels.cpu(), preds.cpu())
