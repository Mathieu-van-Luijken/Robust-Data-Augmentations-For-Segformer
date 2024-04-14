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
                                    transforms.ConvertImageDtype(torch.float32),
                                    transforms.Resize((1024,1024))])(image)

        target = transforms.Compose([transforms.PILToTensor(),
                                     transforms.Resize((1024,1024))])(target)       
        return image, target
    
    def image_to_tensor_test(self, image, target):
        image = transforms.Compose([transforms.PILToTensor(),
                                    transforms.ConvertImageDtype(torch.float32),])(image)

        target = transforms.Compose([transforms.PILToTensor(),
                                     transforms.Resize((1024,512)), 
                                     transforms.CenterCrop((512,512))])(target)       
        return image, target
        

    def load_train_data(self, args):
        """
        """
        train_data = Cityscapes(root=Path(args.data_path), transforms=self.image_to_tensor, split='train', mode='fine', target_type='semantic')
        val_data = Cityscapes(root=Path(args.data_path), transforms=self.image_to_tensor, split='train', mode='fine', target_type='semantic')
 
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        return train_loader, val_loader
 
    def load_eval_data(self, args, data_path):
        test_data = Cityscapes(root=data_path, transforms=self.image_to_tensor_test, mode='fine', target_type='semantic')
        test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch, num_workers=args.num_workers) 

        return test_loader
        
