import torch
import torchvision
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms 

from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader, random_split
from pathlib import Path


import numpy as np

class CityscapesDataLoader():
    def __init__(self):
        pass
        
    
    def image_to_tensor(self, image, target):
        image = transforms.Compose([transforms.PILToTensor(),
                                    transforms.ConvertImageDtype(torch.float32)])(image)

        target = transforms.Compose([transforms.PILToTensor()])(target)       
        return image, target

        

    def load_data(self, args):
        """
        """
        train_data = Cityscapes(root=Path(args.data_path), transforms=self.image_to_tensor, split='train', mode='fine', target_type='semantic')
        val_data = Cityscapes(root=Path(args.data_path), transforms=self.image_to_tensor, split='train', mode='fine', target_type='semantic')
 
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        return train_loader, val_loader
        