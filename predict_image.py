# predict_image.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\resnet18_deepfake.pth"

# MUST MATCH training transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def predict(path, threshold=0.5):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [p_real, p_fake]
        p_real, p_fake = float(probs[0]), float(probs[1])
        pred_idx = 1 if p_fake >= threshold else 0
        pred_label = "FAKE" if pred_idx == 1 else "REAL"
        return pred_label, p_real, p_fake

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path> [threshold]")
        sys.exit(1)
    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    model = load_model()
    label, p_real, p_fake = predict(image_path, threshold)
    print(f"Prediction: {label} | p(REAL)={p_real:.3f} | p(FAKE)={p_fake:.3f} | threshold={threshold}")
