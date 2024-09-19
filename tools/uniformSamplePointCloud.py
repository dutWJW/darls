import numpy as np
from utils.uniformSample.build import uniformSample

def UniformSamplePointCloud(pc1, pc2, gridSize = 1):
    pc1Min = np.min(pc1, axis = 0).astype(np.float32)
    pc1Max = np.max(pc1, axis = 0).astype(np.float32)
    retUniformSamplePc1Index = uniformSample.uniformSample(pc1, pc1Min, pc1Max, gridSize)
    pc2Min = np.min(pc2, axis = 0).astype(np.float32)
    pc2Max = np.max(pc2, axis = 0).astype(np.float32)
    retUniformSamplePc2Index = uniformSample.uniformSample(pc2, pc2Min, pc2Max, gridSize)

    return retUniformSamplePc1Index, retUniformSamplePc2Index
