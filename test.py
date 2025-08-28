# test.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys

# ==========================
# 1. Device setup
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# 2. Load trained model
# ==========================
model = models.resnet18(weights=None)   # Don't use pretrained here
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)   # 2 classes: REAL or FAKE

model.load_state_dict(torch.load("resnet18_deepfake.pth", map_location=device))
model.to(device)
model.eval()

# ==========================
# 3. Define image transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================
# 4. Predict function
# ==========================
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prob = torch.softmax(outputs, dim=1)[0][predicted].item()

    classes = ["REAL", "FAKE"]
    return classes[predicted], prob

# ==========================
# 5. Run from CLI
# ==========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    label, confidence = predict(image_path)
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
