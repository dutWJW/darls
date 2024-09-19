import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from models.SparseConvolution import SparseConvolution
import torch.nn as nn

class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.model = SparseConvolution().to('cuda:0')

    def forward(self, x):
        x = self.model(x)
        return x