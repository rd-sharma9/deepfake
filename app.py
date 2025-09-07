# ap

import streamlit as st

try:
    # Your existing app code starts here
    
    st.title("My Deepfake Detector")
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        # More code for your detection...
    
    # Continue with the rest of your app

except Exception as e:
    st.error(f"Error: {e}")





import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =====================
# CONFIG
# =====================
MODEL_PATH = "C:/Users/ASUS/OneDrive/Desktop/deepfake-detector/resnet18_deepfake.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# =====================
# IMAGE TRANSFORMS
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# STREAMLIT UI
# =====================
st.title("🕵️ Deepfake Detector")
st.write("Upload an image and let the model predict if it's **FAKE** or **REAL**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100

    # Show result
    if pred == 1:
        st.error(f"🚨 FAKE detected with {confidence:.2f}% confidence")
    else:
        st.success(f"✅ REAL detected with {confidence:.2f}% confidence")
