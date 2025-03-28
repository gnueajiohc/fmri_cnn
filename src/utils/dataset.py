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
    """
    Flanker Dataset
    
    Args:
        csv_file (str): path to csv file
        image_dir (str): path to image directory
        width (int): width of image
        view_index (int): select specific view 
        transform (Transforms): apply transform
    """
    def __init__(self, csv_file, image_dir, width, view_index, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.width = width
        self.view_index = view_index
    
    def __len__(self):
        """ return length """
        return len(self.data)
    
    def __getitem__(self, index):
        """ return specific index """
        row = self.data.iloc[index]
        sample_id, label = row["sample_id"], row["label"]
        views = ["axial", "coronal", "sagittal"]
        
        if self.view_index is None: # use all 3 views in training
            images = []
            for view in views:
                path = os.path.join(self.image_dir, f"{sample_id}_{view}.png")
                img = Image.open(path).convert('L')
                if self.transform:
                    img = self.transform(img)
                images.append(img)

            input_tensor = torch.cat(images, dim=0)  # [3, width, width]
        else:                       # use specific one of 3 views
            view = views[self.view_index]
            path = os.path.join(self.image_dir, f"{sample_id}_{view}.png")
            img = Image.open(path).convert('L')
            if self.transform:
                img = self.transform(img)
            input_tensor = img # [1, width, width]
            
        return input_tensor, label

def get_dataloader(batch_size, train, shuffle=True, width=256, seed=42, view_index=None):
    """
    Get train_loader, test_loader from the dataset
    
    Args:
        batch_size (int): batch size
        train (bool): train dataset or test dataset
        shuffle (bool): shuffle dataset if True
        width (int): width of image
        seed (int): random seed
        view_index (int): select specific view 

    Returns:
        dataloader (DataLoader): data loader
    """
    # transform into (width x width)
    transform = transforms.Compose([
        transforms.Resize((width, width)),
        transforms.ToTensor(),
    ])
    
    # seed
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    if train is True:
        dataset = FlankerDataset(csv_file=TRAIN_CSV,
                                 image_dir=IMAGE_DIR,
                                 width=width,
                                 view_index=view_index,
                                 transform=transform)
    else:
        dataset = FlankerDataset(csv_file=TEST_CSV,
                                 image_dir=IMAGE_DIR,
                                 width=width,
                                 view_index=view_index,
                                 transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
    return dataloader
