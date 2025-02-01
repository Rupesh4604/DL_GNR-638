import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from get_image_paths import get_image_paths
from get_bags_of_sifts import get_bags_of_sifts


DATA_PATH = '/Assignment1_Scene-Recognition-with-Bag-of-Words/data/'

CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential',
              'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
              'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = ['agr', 'pln', 'bbd', 'bch', 'bld', 'chp', 'drs',
                   'for', 'frw', 'gof', 'hrb', 'int', 'mrs', 'mhp',
                   'ops', 'pkb', 'riv', 'rwy', 'srs', 'stg', 'tns']

train_image_paths, test_image_paths, val_image_paths, train_labels, test_labels, val_labels = get_image_paths(DATA_PATH, CATEGORIES)

# Load your data (Assuming train_image_feats is (N, 128) and train_labels is (N,))
if os.path.isfile('train_image_feats_1.pkl') is False:
    train_image_feats = get_bags_of_sifts(train_image_paths);
    with open('train_image_feats_1.pkl', 'wb') as handle:
        pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('train_image_feats_1.pkl', 'rb') as handle:
            train_image_feats = pickle.load(handle)

if os.path.isfile('test_image_feats_1.pkl') is False:
    test_image_feats  = get_bags_of_sifts(test_image_paths);
    with open('test_image_feats_1.pkl', 'wb') as handle:
        pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('test_image_feats_1.pkl', 'rb') as handle:
        test_image_feats = pickle.load(handle)

X = train_image_feats  # Features (Bag-of-Words representation, size 128)
y = train_labels       # Labels (Class labels)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size=128, hidden_size1=64, hidden_size2=32, num_classes=21):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation before softmax
        return self.softmax(x)

# Instantiate the model
model = MLP(input_size=128, hidden_size1=128, hidden_size2=64, num_classes=len(set(y.numpy())))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    val_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), "mlp_scene_recognition.pth")
