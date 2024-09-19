import numpy as np
from utils.distributionSample.build import distributionSample

def GenerateSimilarPcBasedDistribution(pc1, pc2, gridSize = 1):
    pc1And2 = np.concatenate([pc1, pc2], axis = 0)
    pc1And2Min = np.min(pc1And2, axis = 0)
    pc1And2Max = np.max(pc1And2, axis = 0)
    (pc1_, pc2_) = distributionSample.distributionSample(pc1, pc2, pc1And2Min, pc1And2Max, gridSize)
    return pc1_, pc2_
