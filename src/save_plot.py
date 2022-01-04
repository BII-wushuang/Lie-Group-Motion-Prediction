#############################################################################
# Input: 3D joint locations
# Plot out the animated motion
#############################################################################
import numpy as np
import matplotlib
import scipy.io as sio
import os
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

colors = {'gt': 'black', 'cem': '#2F4F4F', 'resgru': '#2bba00', 'erd': '#5e01a0', 'srnn': '#0780ea', 'quaternet': '#e08302', 'dmgnn': '#BC8F8F', 'hmr': '#f94e3e', 'xyz': '#008080'}

xmax = 640.32
xmin = -422.06
ymax = 561.78
ymin = -552.46
zmax = 655.77
zmin = -968.78


for action in ['walking']:
    # for method in ['gt', 'erd', 'srnn', 'resgru', 'quaternet', 'cem', 'dmgnn', 'hmr']:
    for method in ['resgru']:
        for k in range(1):
            predict = sio.loadmat('Results/H3.6m/{}/{}_{}.mat'.format(method, action, k))
            predict = predict[list(predict.keys())[3]]
            fig = plt.figure()

            nframes = predict.shape[0]
            lns = []
            for j in range(nframes):
                joint_xyz = np.squeeze(predict[j, :, :])
                x_data = joint_xyz[:, 0]
                y_data = joint_xyz[:, 1]
                z_data = joint_xyz[:, 2]

                ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
                ax.set_axis_off()

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
                         np.array([13, 17, 18, 19, 22]),
                         np.array([13, 25, 26, 27, 30])]
                for i in range(len(chain)):
                    lns.append(ax.plot3D(x_data[chain[i][:],], y_data[chain[i][:],], z_data[chain[i][:],], linewidth=3.0, color=colors[method]))

                plt.tight_layout()

                if not (os.path.exists('Results/H3.6m/{}/{}'.format(method, action))):
                    os.makedirs('Results/H3.6m/{}/{}'.format(method, action))
                plt.savefig('Results/H3.6m/{}/{}/{}_{}.png'.format(method, action, k, j), bbox_inches='tight', pad_inches=0)