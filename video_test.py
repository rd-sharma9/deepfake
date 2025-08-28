import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import sys
import numpy as np

# ----------------------------
# Load Trained Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes: REAL / FAKE
model.load_state_dict(torch.load("resnet18_deepfake.pth", map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# Transform for frames
# ----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ----------------------------
# Predict function
# ----------------------------
def predict_frame(frame):
    img = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item()

# ----------------------------
# Process Video
# ----------------------------
def process_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    predictions = []
    confidences = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:  # Take every nth frame
            pred, conf = predict_frame(frame)
            predictions.append(pred)
            confidences.append(conf)

        frame_idx += 1

    cap.release()

    if len(predictions) == 0:
        print("No frames processed!")
        return

    # Majority voting
    final_pred = 1 if predictions.count(1) > predictions.count(0) else 0
    avg_conf = np.mean(confidences)

    label = "REAL" if final_pred == 0 else "FAKE"
    print(f"Final Prediction: {label} (Confidence: {avg_conf:.2f})")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python video_test.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    process_video(video_path)
