from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms
import os

class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

        # Map string labels to integers
        self.label_map = {"REAL": 0, "FAKE": 1}

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # ResNet input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label_str = self.data.iloc[idx, 1]  # 'FAKE' or 'REAL'
        label = self.label_map[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label
