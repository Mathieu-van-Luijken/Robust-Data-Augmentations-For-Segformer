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

def preprocess(img):
    image = transforms.Compose([transforms.PILToTensor(),
                                transforms.ConvertImageDtype(torch.float32),
                                transforms.Resize((1024,1024))])(img)
    return image.unsqueeze(0)

def postprocess(prediction, shape):
    upsampled_logits = F.interpolate(prediction, size=(shape[0], shape[1]), mode='bilinear', align_corners=False)

    prediction  = torch.argmax(input=upsampled_logits, dim=1)
    output_image_matrix = prediction.cpu().detach().numpy()
    output_image_array = np.transpose(output_image_matrix, (1,2,0))
    return output_image_array