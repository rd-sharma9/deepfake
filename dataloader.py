import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import DeepfakeDataset

# Define transforms (same as before)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Paths
train_csv = r"D:\deepfake-detector\train.csv"
val_csv   = r"D:\deepfake-detector\val.csv"
test_csv  = r"D:\deepfake-detector\test.csv"
root_dir  = r"D:\deepfake-detector\data\frames"

# Create datasets
train_dataset = DeepfakeDataset(train_csv, root_dir, transform=transform)
val_dataset   = DeepfakeDataset(val_csv, root_dir, transform=transform)
test_dataset  = DeepfakeDataset(test_csv, root_dir, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Example: fetch one batch
if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)   # [32, 3, 224, 224]
    print("Labels:", labels[:10])
