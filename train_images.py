import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
checkpoint_path = "deepfake_checkpoint.pth"
num_epochs = 10
batch_size = 32
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# DATA
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

data_dir = r"D:\deepfake-detector\data"   # <-- your dataset path
train_dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"Total images: {len(train_dataset)}")
print(f"Classes: {train_dataset.classes}")

# -------------------------
# MODEL
# -------------------------
model = models.resnet18(weights="IMAGENET1K_V1")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -------------------------
# LOAD CHECKPOINT (if exists)
# -------------------------
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}")

# -------------------------
# TRAIN LOOP
# -------------------------
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        loop.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2f}%"})

    # -------------------------
    # SAVE CHECKPOINT
    # -------------------------
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, checkpoint_path)

    print(f"✅ Epoch {epoch+1} finished | Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f}% | Saved checkpoint")

print("🎉 Training complete!")
