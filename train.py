import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
import os
from tqdm import tqdm  # progress bar

# ==============================
# Config
# ==============================
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Paths
# ==============================
CSV_DIR = r"D:\deepfake-detector"
TRAIN_CSV = os.path.join(CSV_DIR, "train.csv")
VAL_CSV = os.path.join(CSV_DIR, "val.csv")
ROOT_DIR = r"D:\deepfake-detector\frames"  # <-- change to your frames folder

# ==============================
# Load Data
# ==============================
train_dataset = DeepfakeDataset(TRAIN_CSV, ROOT_DIR)
val_dataset = DeepfakeDataset(VAL_CSV, ROOT_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# Model (ResNet18)
# ==============================
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: FAKE, REAL
model = model.to(DEVICE)

# ==============================
# Loss & Optimizer
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==============================
# Training Loop with Progress Bar
# ==============================
for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    running_loss, correct, total = 0, 0, 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]", unit="batch")

    for imgs, labels in progress_bar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / (len(progress_bar))

        progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{train_acc:.2f}%"})

    # ---- Validation ----
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]", unit="batch"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)

    print(f"\n✅ Epoch [{epoch+1}/{EPOCHS}] Finished | "
          f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}% "
          f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")

# ==============================
# Save Model
# ==============================
torch.save(model.state_dict(), "resnet18_deepfake.pth")
print("✅ Model saved as resnet18_deepfake.pth")
