import torch
import torchvision
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms 
from skimage.segmentation import slic

from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from PIL import Image

import numpy as np

from visualization import *



class CityscapesDataLoader():
    def __init__(self):
        self.seed = 2504
        
    def set_seed(self, args):
        self.seed = args.seed
    
    def set_base_dataset(self, args):
        self.dataset = Cityscapes(root=Path(args.data_path), split='train', mode='fine', target_type='semantic')
    

    def image_to_tensor(self, image, target):
        image = transforms.Compose([transforms.PILToTensor(),
                                    transforms.ConvertImageDtype(torch.float32),
                                    transforms.Resize((1024,2048))])(image)

        target = transforms.Compose([transforms.PILToTensor(),
                                     ])(target)       
        return image, target
    
    def basic_augmentations(self, image, target):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(1024, 2048), scale=(0.5, 2.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])
    
        random.seed(self.seed)
        image = transform(image)
        random.seed(self.seed)
        target = transform(target)
        
        return image, target
    

    def blur_augmentations(self, image, target):
        # Define transformations
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(1024, 2048), scale=(0.5, 2.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])

        # Apply the same transformation to both image and target

        random.seed(self.seed)
        image = transform(image)
        random.seed(self.seed)  # Reset the seed to ensure the same random operations
        target = transform(target)

        # Apply Gaussian noise to both image and target
        image = self.apply_gaussian_noise(image)
        target = self.apply_gaussian_noise(target)

        return image, target

    def apply_gaussian_noise(self, image):
        # Add Gaussian noise to the image
        noise = torch.randn_like(image)
        noisy_image = image + noise
        return noisy_image


    def image_to_tensor_test(self, image, target):
        image = transforms.Compose([transforms.PILToTensor(),
                                    transforms.ConvertImageDtype(torch.float32), ])(image)

        target = transforms.Compose([transforms.PILToTensor()])(target)       
        return image, target
    
    def super_pixel_augmentations(self, image, target):
        random.seed(self.seed)
        instance = np.random.choice(len(self.dataset))

        image2 = self.dataset[instance][0]
        target2 = self.dataset[instance][1]

        # Convert PIL image to numpy array
        image_np = np.array(image)
        image2_np = np.array(image2)

        target_np = np.array(target)
        target2_np = np.array(target2)

        # Perform SLIC superpixel segmentation
        segments = slic(image_np, n_segments=100, compactness=10)

        # Randomly select two segments
        segment_ids = np.unique(segments)
        size = np.ceil(0.6*len(segment_ids)).astype(int)
        segment1_id = np.random.choice(segment_ids, size=size, replace=False)

        # Create mask for each segment
        mask = np.zeros_like(segments)
        mask[np.isin(segments, segment1_id)] = 1
        inverted_mask = 1 - mask

        combined_image = np.zeros_like(image_np)
        combined_image[mask==1] = image_np[mask==1]
        combined_image[inverted_mask==1] = image2_np[inverted_mask==1]


        combined_target = np.zeros_like(target_np)
        combined_target[mask==1] = target_np[mask==1]
        combined_target[inverted_mask==1] = target2_np[inverted_mask==1]
       
        # Convert blended image back to PIL image
        blended_image_pil = Image.fromarray(combined_image)
        blended_target_pil = Image.fromarray(combined_target)


        transform = transforms.Compose([transforms.PILToTensor(),
                            transforms.ConvertImageDtype(torch.float32)])
        

        image = transform(blended_image_pil)
        target = transform(blended_target_pil)

        return image, target

    def load_train_data(self, args):
        """
        """
        self.set_seed(args)
        if args.augmentation == "no":
            train_data = Cityscapes(root=Path(args.data_path), transforms=self.image_to_tensor, split='train', mode='fine', target_type='semantic')
            val_data = Cityscapes(root=Path(args.data_path), transforms=self.image_to_tensor, split='train', mode='fine', target_type='semantic')
    
            train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        elif args.augmentation == 'basic':
            train_data = Cityscapes(root=Path(args.data_path), transforms=self.basic_augmentations, split='train', mode='fine', target_type='semantic')
            val_data = Cityscapes(root=Path(args.data_path), transforms=self.image_to_tensor, split='train', mode='fine', target_type='semantic')
    
            train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
        elif args.augmentation == 'blur':
            train_data = Cityscapes(root=Path(args.data_path), transforms=self.blur_augmentations, split='train', mode='fine', target_type='semantic')
            val_data = Cityscapes(root=Path(args.data_path), transforms=self.image_to_tensor, split='train', mode='fine', target_type='semantic')
    
            train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        elif args.augmentation == 'superpixel':
            self.set_base_dataset(args=args)
            train_data = Cityscapes(root=Path(args.data_path), transforms=self.super_pixel_augmentations, split='train', mode='fine', target_type='semantic')
            val_data = Cityscapes(root=Path(args.data_path), transforms=self.image_to_tensor, split='train', mode='fine', target_type='semantic')
    
            train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        return train_loader, val_loader
 
    def load_eval_data(self, args, data_path):
        test_data = Cityscapes(root=data_path, transforms=self.image_to_tensor_test, mode='fine', target_type='semantic')
        test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch, num_workers=args.num_workers) 

        return test_loader
        
