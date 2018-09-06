#############################################################################
# Input: 3D joint locations
# Plot out the animated motion
#############################################################################
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_animation(predict, labels, config, filename):
    if config.dataset == 'Human':
        if config.datatype == 'xyz':
            predict_plot = plot_human(predict, labels, config)
        else:
            predict_plot = plot_h36m(predict, labels, config, filename)
    elif config.dataset == 'Mouse':
        predict_plot = plot_mouse(predict, labels, config)
    elif config.dataset == 'Fish':
        predict_plot = plot_fish(predict, labels, config)
    return predict_plot


class plot_h36m(object):

    def __init__(self, predict, labels, config, filename):
        self.joint_xyz = labels
        self.nframes = labels.shape[0]
        self.joint_xyz_f = predict

        # set up the axes
        xmin = -750
        xmax = 750
        ymin = -750
        ymax = 750
        zmin = -750
        zmax = 750

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self.chain = [np.array([0, 1, 2, 3, 4, 5]),
                      np.array([0, 6, 7, 8, 9, 10]),
                      np.array([0, 12, 13, 14, 15]),
                      np.array([13, 17, 18, 19, 22, 19, 21]),
                      np.array([13, 25, 26, 27, 30, 27, 29])]
        self.scats = []
        self.lns = []
        self.filename = filename

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)

        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])

        xdata_f = np.squeeze(self.joint_xyz_f[frame, :, 0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame, :, 1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame, :, 2])

        for i in range(len(self.chain)):
            self.lns.append(self.ax.plot3D(xdata_f[self.chain[i][:],], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],], linewidth=2.0, color='#f94e3e')) # red: prediction
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],], linewidth=2.0, color='#0780ea')) # blue: ground truth

    def plot(self):
        ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40, repeat=False)
        plt.title(self.filename, fontsize=16)
        #ani.save(self.filename + '.gif', writer='imagemagick')
        plt.show()


class plot_human(object):

    def __init__(self, predict, labels, config):
        self.joint_xyz = labels
        self.nframes = labels.shape[0]
        self.joint_xyz_f = predict

        # set up the axes
        xmin = -1000
        xmax = 1000
        ymin = -1000
        ymax = 1000
        zmin = -1000
        zmax = 1000

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        # self.chain = [np.arange(0, 6), np.arange(6, 12), np.arange(12, 17), np.arange(17, 22), np.arange(22, 27)]
        self.chain = config.chain_idx
        self.scats = []
        self.lns = []

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)

        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])

        xdata_f = np.squeeze(self.joint_xyz_f[frame, :, 0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame, :, 1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame, :, 2])

        # self.scats.append(self.ax.scatter3D(xdata, ydata, zdata, color='b'))
        
        for i in range(len(self.chain)):
            self.lns.append(self.ax.plot3D(xdata_f[self.chain[i][:], ], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],], linewidth=2.0, color='#f94e3e'))
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:], ], ydata[self.chain[i][:], ], zdata[self.chain[i][:],], linewidth=2.0, color='#0780ea'))

    def plot(self):
        ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40)
        plt.show()


class plot_mouse(object):

    def __init__(self, predict, labels, config):
        self.joint_xyz = labels
        self.nframes = labels.shape[0]
        self.joint_xyz_f = predict

        # set up the axes
        xmin = -100
        xmax = 100
        ymin = -100
        ymax = 100
        zmin = -100
        zmax = 100

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        # self.chain = [np.arange(0, 5)]
        self.chain = config.chain_idx
        self.scats = []
        self.lns = []

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)

        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])

        xdata_f = np.squeeze(self.joint_xyz_f[frame,:,0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame,:,1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame,:,2])

        # self.scats.append(self.ax.scatter3D(xdata, ydata, zdata, color='b'))

        for i in range(len(self.chain)):
            self.lns.append(self.ax.plot3D(xdata_f[self.chain[i][:],], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],], linewidth=2.0, color='#f94e3e'))
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],], linewidth=2.0, color='#0780ea'))

    def plot(self):
        ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40)
        plt.show()


class plot_fish(object):

    def __init__(self, predict, labels, config):
        self.joint_xyz = labels
        self.nframes = labels.shape[0]
        self.joint_xyz_f = predict

        # set up the axes
        xmin = -1000
        xmax = 1000
        ymin = -1000
        ymax = 1000
        zmin = -1000
        zmax = 1000

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        # self.chain = [np.arange(0, 21)]
        self.chain = config.chain_idx
        self.scats = []
        self.lns = []

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)

        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])

        xdata_f = np.squeeze(self.joint_xyz_f[frame,:,0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame,:,1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame,:,2])

        # self.scats.append(self.ax.scatter3D(xdata, ydata, zdata, color='b'))

        for i in range(len(self.chain)):
            self.lns.append(self.ax.plot3D(xdata_f[self.chain[i][:],], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],], linewidth=2.0, color='#f94e3e'))
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],], linewidth=2.0, color='#0780ea'))

    def plot(self):
        ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40)
        plt.show()
