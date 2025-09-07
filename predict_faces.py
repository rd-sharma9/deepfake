import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ✅ Paths
MODEL_PATH = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\model_faces.pth"

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Load model
model = models.mobilenet_v2(weights=None)  # lightweight model
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes: real vs fake
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ✅ Predict function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    classes = ["REAL", "FAKE"]
    pred_idx = probs.argmax()
    return classes[pred_idx], probs[0], probs[1]

# ✅ Main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Usage: python predict_faces.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    label, p_real, p_fake = predict(image_path)
    print(f"Prediction: {label} | p(REAL)={p_real:.3f} | p(FAKE)={p_fake:.3f}")
