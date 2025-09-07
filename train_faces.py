import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ✅ Path to your cropped faces dataset
DATA_DIR = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\data\faces"
MODEL_PATH = r"C:\Users\ASUS\OneDrive\Desktop\deepfake-detector\model_faces.pth"

# ✅ Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Load dataset
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# ⚡ Use a small subset for testing (first 1000 images)
train_dataset = Subset(full_dataset, range(1000))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# ✅ Model (MobileNetV2 is lighter than ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 classes
model = model.to(device)

# ✅ Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ✅ Training loop (short test run: 2 epochs)
EPOCHS = 2
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    print(f"✅ Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {100.*correct/total:.2f}%")

# ✅ Save trained model
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
