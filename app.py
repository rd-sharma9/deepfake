import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import traceback
import gdown
import os
from pathlib import Path

# =====================
# CONFIG
# =====================
MODEL_PATH = "resnet18_deepfake.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# =====================
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1Sd6LdJDo9yQFkiG-oJgcpY6-Eza3c7nR"
        gdown.download(url, MODEL_PATH, quiet=False)

download_model()

# =====================
# MAIN FUNCTION
# =====================
def main():
    st.title("🕵️ Deepfake Detector")

    # Image upload section only
    st.write("Upload an image and let the model predict if it's **FAKE** or **REAL**.")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # =====================
    # MODEL LOADING
    # =====================
    @st.cache_resource
    def load_model():
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
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

    # Prediction
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
# RUN APP WITH ERROR HANDLING
# =====================
try:
    main()
except Exception:
    st.error("An unexpected error occurred!")
    st.text(traceback.format_exc())
