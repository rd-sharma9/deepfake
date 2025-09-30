import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown
import zipfile
from pathlib import Path
import os
import traceback

# =====================
# CONFIG
# =====================
MODEL_ID = "1GeFrm3CTFg_N158GODCFQf4iKcsh11NH"   # model_faces.pth
FACES_ID = "1ddUZs_vINTqPHFf2S7vq8RlugJLq6dmn"   # faces.zip.zip
MODEL_PATH = Path("resnet18_deepfake.pth")        # <-- converted to Path
FACES_PATH = Path("data/faces")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# DOWNLOAD ASSETS
# =====================
def download_assets():
    # Download model if not exists
    if not MODEL_PATH.exists():
        st.info("Downloading model file...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", str(MODEL_PATH), quiet=False)

    # Download and unzip faces if not exists
    if not FACES_PATH.exists():
        st.info("Downloading faces dataset...")
        faces_zip = Path("faces.zip")
        gdown.download(f"https://drive.google.com/uc?id={FACES_ID}", str(faces_zip), quiet=False)

        with zipfile.ZipFile(faces_zip, "r") as zip_ref:
            zip_ref.extractall(FACES_PATH.parent)
        faces_zip.unlink()  # remove zip file

download_assets()

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)   # create ResNet18
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes (real/fake)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# MAIN APP
# =====================
def main():
    st.title("🕵️ Deepfake Detector")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item() * 100

        if pred == 1:
            st.error(f"🚨 FAKE detected with {confidence:.2f}% confidence")
        else:
            st.success(f"✅ REAL detected with {confidence:.2f}% confidence")

# =====================
# RUN
# =====================
try:
    main()
except Exception:
    st.error("An unexpected error occurred!")
    st.text(traceback.format_exc())
