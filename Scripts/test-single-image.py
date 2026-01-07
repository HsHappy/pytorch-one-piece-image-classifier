# ============================================================
# FILE: predict_single_image.py
#
# PURPOSE:
# Run inference on individual images using a trained CNN model.
# This script is intended for manual testing and qualitative
# evaluation (no training, no metrics).
# ============================================================

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"C:\Users\Taha\Desktop\okul\veri bilimi 3.1\final projes\main\model\model1.pth"
HISTORY_PATH = r"C:\Users\Taha\Desktop\okul\veri bilimi 3.1\final projes\main\model\training_history.pt"

# CHANGE THIS TO ANY IMAGE YOU WANT TO TEST
IMAGE_PATH = r"C:\Users\Taha\Desktop\okul\veri bilimi 3.1\final projes\main\OnePieceDataset\train\Kurohige\Copied-Image73(Copied-Image164).png"

# ============================================================
# 2. MODEL DEFINITION (MUST MATCH TRAINING SCRIPT)
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

# ============================================================
# 3. LOAD MODEL & CLASS NAMES
# ============================================================

history = torch.load(HISTORY_PATH)
class_names = history["classes"]

model = BasitCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded successfully.")
print("Classes:", class_names)

# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================
# IMPORTANT:
# Must match test-time preprocessing (NO augmentation)

transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image = Image.open(IMAGE_PATH)
input_tensor = transform(image).unsqueeze(0).to(device)

# ============================================================
# 5. INFERENCE
# ============================================================

with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)

predicted_class = class_names[predicted_idx.item()]
confidence = confidence.item() * 100

# ============================================================
# 6. RESULTS
# ============================================================

print("\nPrediction Result")
print("-----------------")
print(f"Image: {IMAGE_PATH}")
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

# ============================================================
# 7. VISUALIZATION (OPTIONAL BUT NICE)
# ============================================================

plt.imshow(image)
plt.axis("off")
plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
plt.show()
