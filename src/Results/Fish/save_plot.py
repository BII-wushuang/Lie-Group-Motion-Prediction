#############################################################################
# Input: 3D joint locations
# Plot out the animated motion
#############################################################################
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import scipy.io as sio
import os
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


files = ['gt', 'erd', 'resgru', 'hmr']

xmax = 2000
xmin = -2000
ymax = 2000
ymin = -2000
zmax = 1500
zmin = -1500

chain = np.arange(21)
colors = ['black', '#5e01a0', '#2bba00', '#f94e3e']
scattersize = np.zeros(21)
for i in range(5):
    scattersize[i] = 180 - i*30
for i in range(5,21):
    scattersize[i] = 30 - (i-5) * 2

frames = [5, 25, 45, 65, 85]

for file in files:
    predict = sio.loadmat('./' + file + '.mat')
    predict = predict[list(predict.keys())[3]]

    fig = plt.figure()

    nframes = predict.shape[0]
    predict = np.reshape(predict,[nframes,-1,3])

    scats = []
    lns = []
    for j in frames:

        joint_xyz = np.squeeze(predict[j, :, :])
        x_data = joint_xyz[:, 0]
        y_data = joint_xyz[:, 1]
        z_data = joint_xyz[:, 2]

        ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        ax.set_axis_off()

        ax.view_init(azim=10, elev=60)
        plt.gca().set_aspect('equal', adjustable='box')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])

        lns.append(ax.plot3D(x_data[chain[:]], y_data[chain[:]], z_data[chain[:]], linewidth=1.0, color=colors[files.index(file)]))
        for k in range(21):
            scats.append(ax.scatter3D(x_data[chain[k]], y_data[chain[k]], z_data[chain[k]], color=colors[files.index(file)], s=scattersize[k], alpha=1.0-k*0.03))

        plt.tight_layout()

        if not (os.path.exists('./' + file)):
            os.makedirs('./' + file)
        plt.savefig('./' + file + '/output_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
