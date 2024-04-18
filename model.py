import torch
import torch.nn as nn

from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn.functional as F

""" segmentation model example
"""

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transfer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b1", num_labels=19)


    def forward(self, inputs):
        logits = self.transfer_model(inputs).logits
        return logits
    
