from tools.o3d_tools import *
import argparse
import glob, os, torch
from models.FeatureExtraction import FeatureExtraction
from models.networkUtils import voxelization
from tools.regisTwoPcWithFeature import Registration
from tools.metric import translation_error, rotation_error
from tqdm import tqdm

def get_files(path, extension):
    return glob.glob(os.path.join(path, f'*{extension}'))

def calTeAndRe(T1, T2):
    return translation_error(T1[:3, 3], T2[:3, 3]), rotation_error(T1[:3, :3], T2[:3, :3])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--dataset', type=str, default='') # KITTI Dataset Path
    parser.add_argument('--model', type=str, default='./models/checkpoints/kitti.pth')
    parser.add_argument('--rteThresh', type=float, default=0.1)
    parser.add_argument('--rreThresh', type=float, default=0.5)

    args = parser.parse_args()
    files = get_files(os.path.join(args.dataset, 'icp'), 'npy')
    model = FeatureExtraction()
    model.load_state_dict(torch.load(args.model)['model_state_dict'])
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        states = []
        for f in tqdm(files):
            [seq, s, t] = f.split('/')[-1].split('.')[0].split('_')
            sourceFilePath = os.path.join(args.dataset, seq.zfill(2), s.zfill(6) + '.bin')
            targetFilePath = os.path.join(args.dataset, seq.zfill(2), t.zfill(6) + '.bin')
            sourcePc_ = np.fromfile(sourceFilePath, dtype=np.float64).reshape(-1, 3).astype(np.float32)
            targetPc_ = np.fromfile(targetFilePath, dtype=np.float64).reshape(-1, 3).astype(np.float32)
            poseGt = np.load(f)
            sourcePcTensor, sourcePc = voxelization(sourcePc_, 0.2)
            targetPcTensor, targetPc = voxelization(targetPc_, 0.2)
            f1 = model(sourcePcTensor.cuda()).cpu().detach().numpy()
            f2 = model(targetPcTensor.cuda()).cpu().detach().numpy()
            RT = Registration(sourcePc, f1, targetPc, f2, sourcePc_, targetPc_)
            TE, RE = calTeAndRe(RT, poseGt)
            states.append(np.array([TE < args.rteThresh and RE < args.rreThresh, TE, RE]))
            torch.cuda.empty_cache()

        states = np.array(states)
        Recall = states[:, 0].sum() / states.shape[0]
        TE = states[states[:, 0] == 1, 1].mean()
        RE = states[states[:, 0] == 1, 2].mean()
        print(f'Recall of DGR: {Recall}')
        print(f'TE of DGR: {TE}')
        print(f'RE of DGR: {RE}')
