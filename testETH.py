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

def loadlog(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--dataset', type=str, default='') # ETH Dataset Path
    parser.add_argument('--env', type=list, default= ['gazebo_winter', 'wood_autumn'])
    parser.add_argument('--model', type=str, default='./models/checkpoints/eth.pth')
    parser.add_argument('--rteThresh', type=float, default=0.1)
    parser.add_argument('--rreThresh', type=float, default=0.5)

    args = parser.parse_args()

    model = FeatureExtraction()
    model.load_state_dict(torch.load(args.model, map_location={'cuda:1': 'cuda:0'})['model_state_dict'])

    model = model.cuda()
    model.eval()

    with torch.no_grad():
        states = []
        for e in args.env:
            gt = loadlog(os.path.join(args.dataset, e))
            for f in tqdm(gt):
                [s, t] = f.split('_')
                sourceFilePath = os.path.join(args.dataset, e, f'cloud_bin_{s}.ply')
                targetFilePath = os.path.join(args.dataset, e, f'cloud_bin_{t}.ply')
                sourcePc_ = np.array(o3d.io.read_point_cloud(sourceFilePath).points).astype(np.float32)
                targetPc_ = np.array(o3d.io.read_point_cloud(targetFilePath).points).astype(np.float32)
                poseGt = np.linalg.inv(gt[f])
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
