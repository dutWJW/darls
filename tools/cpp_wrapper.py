import os
import sys
sys.path.append('/'.join(os.path.dirname(__file__).split('/')[:-1]))
import utils.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
    return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
