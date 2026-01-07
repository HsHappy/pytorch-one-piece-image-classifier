# ============================================================
# FILE: test_and_visualize_model.py
#
# PURPOSE:
# - Load a previously trained CNN model
# - Evaluate it on a SEPARATE test dataset
# - Visualize saved training metrics
# - Produce plots suitable for academic reporting
#
# IMPORTANT:
# This script DOES NOT train the model.
# ============================================================

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# 1. CONFIGURATION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 5

MODEL_PATH = r"C:\Users\Taha\Desktop\okul\veri bilimi 3.1\final projes\main\model\model1.pth"
HISTORY_PATH = r"C:\Users\Taha\Desktop\okul\veri bilimi 3.1\final projes\main\model\training_history.pt"
TEST_DATA_PATH = r"C:\Users\Taha\Desktop\okul\veri bilimi 3.1\final projes\main\OnePieceDataset\validate"

# ============================================================
# 2. DATA PREPROCESSING (NO AUGMENTATION!)
# ============================================================
# NOTE FOR REPORT:
# Test data must NOT be augmented to ensure fair evaluation.

test_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder(
    root=TEST_DATA_PATH,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.classes

print("\n=== TEST DATA INFO ===")
print("Classes:", class_names)
print("Number of test images:", len(test_dataset))
print("======================\n")

# ============================================================
# 3. MODEL DEFINITION (MUST MATCH TRAINING SCRIPT!)
# ============================================================

class BasitCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasitCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

model = BasitCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded successfully.\n")

# ============================================================
# 4. MODEL EVALUATION
# ============================================================

all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total

print(f"Test Accuracy: {test_accuracy:.2f}%")

# ============================================================
# 5. CONFUSION MATRIX
# ============================================================
# NOTE FOR REPORT:
# Confusion matrix provides class-level performance insight.

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Test Data)")
plt.grid(False)
plt.show()

# ============================================================
# 6. LOAD & VISUALIZE TRAINING HISTORY
# ============================================================

history = torch.load(HISTORY_PATH)

train_losses = history["loss"]
train_accuracies = history["accuracy"]
epochs = history["epochs"]

# ============================================================
# 7. TRAINING CURVES (FROM SAVED DATA)
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, marker='o')
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# END OF SCRIPT
# ============================================================
