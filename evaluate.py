import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset import DeepfakeDataset
import os

# ✅ Point everything to C: drive
ROOT_DIR = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\data\faces"
TEST_CSV = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\test.csv"
MODEL_PATH = "C:/Users/ASUS/OneDrive/Desktop/deepfake-detector/resnet18_deepfake.pth"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Loader
test_dataset = DeepfakeDataset(csv_file=TEST_CSV, root_dir=ROOT_DIR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = models.resnet18(weights=None)  # no pretrained, using your trained weights
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Evaluation
correct, total = 0, 0
fake_correct, real_correct, fake_total, real_total = 0, 0, 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for label, pred in zip(labels, preds):
            if label == 1:  # FAKE
                fake_total += 1
                if label == pred:
                    fake_correct += 1
            else:  # REAL
                real_total += 1
                if label == pred:
                    real_correct += 1

accuracy = 100 * correct / total if total > 0 else 0
fake_acc = 100 * fake_correct / fake_total if fake_total > 0 else 0
real_acc = 100 * real_correct / real_total if real_total > 0 else 0

print(f"✅ Overall Accuracy: {accuracy:.2f}%")
print(f"🎭 FAKE Accuracy: {fake_acc:.2f}%")
print(f"🙂 REAL Accuracy: {real_acc:.2f}%")



from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# After evaluation loop:
all_preds = []
all_labels = []
all_probs = []   # <-- new

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of FAKE class
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())  # <-- store probabilities


# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"]))

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")   # ✅ Save as PNG
plt.show()


# ---- ROC Curve ----
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")   # ✅ Save as PNG
plt.show()


# ---- Precision-Recall Curve ----
precision, recall, _ = precision_recall_curve(all_labels, all_preds)

plt.figure()
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig("precision_recall_curve.png")   # ✅ Save as PNG
plt.show()




from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Convert labels & predictions to numpy arrays
import numpy as np
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ---- ROC Curve ----
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# ---- Precision-Recall Curve ----
precision, recall, _ = precision_recall_curve(all_labels, all_preds)

plt.figure()
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
