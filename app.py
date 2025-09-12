import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import traceback
import os
import gdown

# =====================
# CONFIG
# =====================
MODEL_PATH = Path("resnet18_deepfake.pth")
DRIVE_ID = "1Sd6LdJDo9yQFkiG-oJgcpY6-Eza3c7nR"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Download model if not present
# =====================
def download_model():
    if not MODEL_PATH.exists():
        st.write("Downloading model…")
        gdown.download(DRIVE_URL, str(MODEL_PATH), quiet=False)

download_model()

# =====================
# Load model
# =====================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =====================
# UI
# =====================
st.title("🕵️ Deepfake Detector")

uploaded_image = st.file_uploader("Upload image…", type=["jpg","jpeg","png"])
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100

    if pred == 1:
        st.error(f"🚨 FAKE ({confidence:.2f}%)")
    else:
        st.success(f"✅ REAL ({confidence:.2f}%)")
