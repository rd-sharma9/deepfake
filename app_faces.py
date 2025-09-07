import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ======================
# 🔹 Load trained model
# ======================
MODEL_PATH = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\model_faces.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define same architecture as training
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, 2)  # binary: real/fake
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

# ======================
# 🔹 Transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================
# 🔹 Prediction function
# ======================
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        p_real, p_fake = probs[0].item(), probs[1].item()

    label = "REAL" if p_real >= p_fake else "FAKE"
    return label, p_real, p_fake

# ======================
# 🔹 Streamlit UI
# ======================
st.title("🕵️ Deepfake Detector (Face Images)")
st.write("Upload a face image and the model will predict if it is **REAL** or **FAKE**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        label, p_real, p_fake = predict_image(image)
        st.write(f"**Prediction:** {label}")
        st.write(f"✅ Probability REAL: {p_real:.3f}")
        st.write(f"❌ Probability FAKE: {p_fake:.3f}")
