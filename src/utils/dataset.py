import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
import os

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent
DATA_DIR = SRC_DIR / "data"
IMAGE_DIR = DATA_DIR / "samples"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

class FlankerDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        sample_id, label = row["sample_id"], row["label"]
        views = ["axial", "coronal", "sagittal"]
        images = []
        
        for view in views:
            path = os.path.join(self.image_dir, f"{sample_id}_{view}.png")
            img = Image.open(path).convert('L')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        input_tensor = torch.cat(images, dim=0)  # [3, width, width]
        return input_tensor, label

def get_dataloader(batch_size, train, shuffle=True, width=256):
    transform = transforms.Compose([
        transforms.Resize((width, width)),
        transforms.ToTensor(),
    ])
    
    if train is True:
        dataset = FlankerDataset(csv_file=TRAIN_CSV,
                                 image_dir=IMAGE_DIR,
                                 transform=transform)
    else:
        dataset = FlankerDataset(csv_file=TEST_CSV,
                                 image_dir=IMAGE_DIR,
                                 transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
