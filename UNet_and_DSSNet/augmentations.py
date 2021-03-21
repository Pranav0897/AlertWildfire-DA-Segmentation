import torch
import torch.nn as nn


class SyntheticAugmentation(nn.Module):

    def __init__(self, args):
        super(SyntheticAugmentation, self).__init__()
    
    def forward(self, data_dict):
        aug_dict = data_dict 
        return aug_dict