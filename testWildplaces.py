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

def getGt(poseFiles):
    gt = {}
    start = 0
    next = 1
    startPose = 0
    for i_, p in enumerate(poseFiles):
        if i_ == 0:
            startPose = np.loadtxt(p)
            continue
        nextPose = np.loadtxt(p)
        gt[str(start) + '_' + str(next)] = np.linalg.inv(nextPose) @ startPose
        startPose = np.loadtxt(p)
        start += 1
        next += 1
    return gt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--dataset', type=str, default='') # wildplaces Dataset Path
    parser.add_argument('--model', type=str, default='./models/checkpoints/wildplaces.pth')
    parser.add_argument('--rteThresh', type=float, default=0.1)
    parser.add_argument('--rreThresh', type=float, default=0.5)

    args = parser.parse_args()
    model = FeatureExtraction()
    model.load_state_dict(torch.load(args.model)['model_state_dict'])
    model = model.cuda()
    model.eval()
    poseFiles = get_files(args.dataset, '.txt')
    poseFiles.sort(key = lambda x : int(x.split('/')[-1].split('_')[0]))
    gt = getGt(poseFiles)

    with torch.no_grad():
        states = []
        for f in tqdm(gt):
            [s, t] = f.split('_')
            sourceFilePath = os.path.join(args.dataset, f'{s}_pc.pcd')
            targetFilePath = os.path.join(args.dataset, f'{t}_pc.pcd')
            sourcePc_ = np.array(o3d.io.read_point_cloud(sourceFilePath).points).astype(np.float32)
            targetPc_ = np.array(o3d.io.read_point_cloud(targetFilePath).points).astype(np.float32)
            poseGt = gt[f]
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
        print('Recall of DGR: {:.2f}'.format(Recall*100))
        print('TE of DGR: {:.2f}'.format(TE*100))
        print('RE of DGR: {:.2f}'.format(RE))
