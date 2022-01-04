from __future__ import print_function, division
import argparse
import torch
import os
import glob
import numpy as np
import scipy.io as sio
import joblib
import smplx
import trimesh
import h5py
from progressbar import progressbar
from smplify import SMPLify3D
import config


def fit_smpl(data):
    """
    fit smpl model with 3d keypoints data
    """
    
    nframes = len(data)

    pose = np.zeros([nframes, 72])
    trans = np.zeros([nframes, 3])
    
    j3d = torch.Tensor(data[0:1]).float()
    _, _, fitted_pose, fitted_beta, fitted_trans, _ = smplify_base(init_mean_pose, init_mean_shape, init_cam_trans, j3d, True, 1.0)
    pose[0] = fitted_pose.numpy()
    trans[0] = fitted_trans.numpy()
    beta = fitted_beta.numpy().squeeze()
    
    for idx in progressbar(range(1, nframes)):
        j3d = torch.Tensor(data[idx:idx+1]).float()
        _, _, fitted_pose, _, fitted_trans, _ = smplify_base(fitted_pose, fitted_beta, fitted_trans, j3d, False, 1.0)
        
        pose[idx] = fitted_pose.numpy()
        trans[idx] = fitted_trans.numpy()
    
    return pose, beta, trans
    

def fit_smpl_batch(data):
    """
    fit smpl model with 3d keypoints data
    """
    
    nframes = len(data)
    
    pose = np.zeros([nframes, 72])
    trans = np.zeros([nframes, 3])
    
    j3d = torch.Tensor(data[0:1]).float()
    _, _, fitted_pose, fitted_beta, fitted_trans, _ = smplify_base(init_mean_pose, init_mean_shape, init_cam_trans, j3d, True, 1.0)
    pose[0] = fitted_pose.numpy()
    trans[0] = fitted_trans.numpy()
    beta = fitted_beta.numpy().squeeze()

    batch_j3d = torch.Tensor(data[-1:]).float().repeat([opt.batchSize, 1, 1]).to(device)
    for idx in progressbar(range(1, nframes, opt.batchSize)):
        n = min(nframes,idx+opt.batchSize) - idx
        batch_j3d[:n] = torch.Tensor(data[idx:idx+n]).float().to(device)

        _, _, fitted_pose, _, fitted_trans, _ = smplify(init_mean_pose.repeat([opt.batchSize,1]).to(device), fitted_beta.repeat([opt.batchSize,1]).to(device), init_cam_trans.repeat([opt.batchSize,1]).to(device), batch_j3d, False, 1.0)
        
        pose[idx:idx+n] = fitted_pose.cpu().numpy()[:n]
        trans[idx:idx+n] = fitted_trans.cpu().numpy()[:n]
    
    return pose, beta, trans


### Main ###

# parsing argmument
parser = argparse.ArgumentParser()
parser.add_argument('--joints_category', type=str, default='NTU', help='choose joints category')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--num_smplify_iters', type=int, default=100, help='num of smplify iters')
parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
parser.add_argument('--gpu_ids', type=int, default=0, help='choose gpu ids')
opt = parser.parse_args()
print(opt)

# device
device = torch.device('cuda:' + str(opt.gpu_ids) if opt.cuda else 'cpu')

# load mean pose
smpl_mean_file = config.SMPL_MEAN_FILE
file = h5py.File(smpl_mean_file, 'r')
init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).float()
init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).float()
init_cam_trans = torch.Tensor([0.0, 0.0, 0.0])

# initialize SMPLify
smplify_base = SMPLify3D(smplxmodel=smplx.create(config.SMPL_MODEL_DIR, model_type='smpl', gender='neutral', ext='pkl', batch_size=1).to('cpu'),
                        batch_size=1,
                        use_collision=False,
                        joints_category=opt.joints_category,
                        device='cpu')

smplify = SMPLify3D(smplxmodel=smplx.create(config.SMPL_MODEL_DIR, model_type='smpl', gender='neutral', ext='pkl', batch_size=opt.batchSize).to(device),
                    batch_size=opt.batchSize,
                    use_collision=False,
                    joints_category=opt.joints_category,
                    device=device)

print('initialize SMPLify3D done!')

folder = './Results/'
save_folder = './Rendered/'
os.makedirs(save_folder, exist_ok = True)

files = glob.glob(folder + '*.npy')

for file in files:
    filename = os.path.split(os.path.splitext(file)[0])[1]
    print('Processing ' + filename)
    data = np.load(file)
    # data = data/1000.0
    
    pose, beta, trans = fit_smpl(data)
    param = {}
    param['beta'] = beta
    param['pose'] = pose
    param['trans'] = trans
    joblib.dump(param, save_folder + filename + '.pkl', compress=3)
