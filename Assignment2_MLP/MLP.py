import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define categories (same as UC Merced)
CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
              'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
              'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']

# Global label encoder for consistency across datasets
label_encoder = LabelEncoder()
label_encoder.fit(CATEGORIES)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image data directory
data_dir = 'data/'

# Custom Dataset for loading and processing images
class ImageDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.images = []
        self.labels = []

        # Loop over categories to collect images
        for label in CATEGORIES:
            category_dir = os.path.join(data_dir, split, label)
            if not os.path.exists(category_dir):  # Skip if directory does not exist
                continue
            for img_name in os.listdir(category_dir):
                if img_name.endswith(".jpg"):  # Assuming the images are in .jpg format
                    img_path = os.path.join(category_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)

        # Convert labels to integers using the global label encoder
        self.labels = label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load and preprocess image
        img_path = self.images[idx]
        img = Image.open(img_path)
        img = img.resize((72, 72))
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
        img = img.flatten()

        # Convert to tensor
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label

# Create datasets
train_dataset = ImageDataset(data_dir, split='train')
val_dataset = ImageDataset(data_dir, split='val')
test_dataset = ImageDataset(data_dir, split='test')

# Create DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size=72*72, hidden_size1=512, hidden_size2=256, num_classes=len(CATEGORIES)):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Instantiate and move the model to the device
model = MLP().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Start Training")

# Best model tracking
best_val_accuracy = 0.0

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        batch_X = batch_X.view(batch_X.size(0), -1)  # Flatten input

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
            batch_X = batch_X.view(batch_X.size(0), -1)  # Flatten input
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

# Evaluate on Test Set
model.load_state_dict(torch.load("mlp_scene_recognition_best.pth"))
model.eval()

correct, total = 0, 0
predictions = []
true_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_X = batch_X.view(batch_X.size(0), -1)  # Flatten input
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(batch_y.cpu().numpy())

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
decoded_predictions = label_encoder.inverse_transform(predictions)
decoded_true_labels = label_encoder.inverse_transform(true_labels)

cm = confusion_matrix(decoded_true_labels, decoded_predictions, labels=CATEGORIES)
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
