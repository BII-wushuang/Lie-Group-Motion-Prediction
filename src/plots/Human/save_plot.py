#############################################################################
# Input: 3D joint locations
# Plot out the animated motion
#############################################################################
import numpy as np
import matplotlib
import scipy.io as sio
import os
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


files = ['gt', '3lr', 'erd', 'resgru', 'xyz', 'hmr']
colors = ['black', '#0780ea', '#5e01a0', '#2bba00', '#e08302', '#f94e3e']
# 3lr: blue: #0780ea; hmr: red: #f94e3e; gt: black, k; resgru: green, #2bba00; erd: purple: #5e01a0; xyz: blue: #e08302

frames = [0, 25, 50, 75, 100, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250]

xmax = 640.32
xmin = -422.06
ymax = 561.78
ymin = -552.46
zmax = 655.77
zmin = -968.78

for file in files:
    predict = sio.loadmat('./walking_' + file +'.mat')
    predict = predict[list(predict.keys())[3]]

    fig = plt.figure()

    nframes = predict.shape[0]
    predict = np.reshape(predict,[nframes,-1,3])

    lns = []
    for j in range(len(frames)):

        joint_xyz = np.squeeze(predict[frames[j], :, :])
        x_data = joint_xyz[:, 0]
        y_data = joint_xyz[:, 1]
        z_data = joint_xyz[:, 2]

        ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')

        ax.view_init(azim=85, elev=5)
        plt.gca().set_aspect('equal', adjustable='box')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])

        chain = [np.array([0, 1, 2, 3, 4, 5]),
                 np.array([0, 6, 7, 8, 9, 10]),
                 np.array([0, 12, 13, 14, 15]),
                 np.array([13, 17, 18, 19, 22, 19, 21]),
                 np.array([13, 25, 26, 27, 30, 27, 29])]
        for i in range(len(chain)):
            lns.append(ax.plot3D(x_data[chain[i][:],], y_data[chain[i][:],], z_data[chain[i][:],], linewidth=3.0, color=colors[files.index(file)]))

        plt.tight_layout()

        if not (os.path.exists('./' + file)):
            os.makedirs('./' + file)
        plt.savefig('./' + file + '/output_' + str(frames[j]) + '.png', bbox_inches='tight', pad_inches=0)
