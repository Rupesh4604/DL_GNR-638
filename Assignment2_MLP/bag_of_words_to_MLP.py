import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define categories
CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
              'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
              'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']
CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load feature data
with open('train_image_feats_1.pkl', 'rb') as handle:
    train_image_feats = pickle.load(handle)

with open('test_image_feats_1.pkl', 'rb') as handle:
    test_image_feats = pickle.load(handle)

with open('val_image_feats_1.pkl', 'rb') as handle:
    val_image_feats = pickle.load(handle)

print(train_image_feats.shape)

# Load labels
path = "labels/"
train_y = np.load("train_labels.npy")
val_y = np.load("val_labels.npy")
test_y = np.load("test_labels.npy")

print("Data Loaded")
# Convert to PyTorch tensors
train_X = torch.tensor(train_image_feats.astype(np.float32))
val_X = torch.tensor(val_image_feats.astype(np.float32))
test_X = torch.tensor(test_image_feats.astype(np.float32))

# Convert labels to integer values based on CATE2ID mapping
train_y = np.array([CATE2ID[label] for label in train_y])  # Convert string labels to integers
val_y = np.array([CATE2ID[label] for label in val_y])  # Convert string labels to integers
test_y = np.array([CATE2ID[label] for label in test_y])  # Convert string labels to integers

train_y = torch.tensor(train_y, dtype=torch.long)
val_y = torch.tensor(val_y, dtype=torch.long)
test_y = torch.tensor(test_y, dtype=torch.long)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)
test_dataset = TensorDataset(test_X, test_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size=150, hidden_size1=128, hidden_size2=64, num_classes=21):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# Instantiate and move the model to the device
model = MLP(input_size=150, hidden_size1=128, hidden_size2=64, num_classes=len(CATEGORIES)).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Start Training")

# Best model tracking
best_val_accuracy = 0.0

# Training loop
num_epochs = 1500
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    val_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "mlp_scene_recognition_best.pth")

# Save the final model (even if it's not the best)
torch.save(model.state_dict(), "mlp_scene_recognition_final.pth")

model.load_state_dict(torch.load("mlp_scene_recognition_best.pth"))
model.eval()

correct, total = 0, 0
predictions = []
true_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(batch_y.cpu().numpy())

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.show()

