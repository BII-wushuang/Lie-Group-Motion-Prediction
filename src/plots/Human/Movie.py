from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy.io as sio


files = ['gt', '3lr', 'erd', 'resgru', 'xyz', 'hmr']
titles = ['Ground Truth', 'LSTM-3LR (2015)', 'ERD (2015)', 'Res-GRU (2017)', 'XYZ Baseline', 'HMR (Ours)']

data = {}
for file in files:
    datafile = sio.loadmat('./walking_' + file + '.mat')
    data[file] = datafile[list(datafile.keys())[3]]


class plot_human(object):

    def __init__(self, data):

        self.data = data
        self.files = list(data.keys())
        self.nframes = data[self.files[0]].shape[0]

        self.chain = [np.array([0, 1, 2, 3, 4, 5]),
                      np.array([0, 6, 7, 8, 9, 10]),
                      np.array([0, 12, 13, 14, 15]),
                      np.array([13, 17, 18, 19, 22, 19, 21]),
                      np.array([13, 25, 26, 27, 30, 27, 29])]

        self.fig = plt.figure()

        self.plt = {}
        plt.title("Long-term forecasting for 'Walking' on H3.6m\n", fontsize=18, fontweight='bold')
        plt.axis('off')
        for i in range(len(self.files)):
            self.plt[self.files[i]] = self.fig.add_subplot(2, 3, i+1, projection='3d')
            self.plt[self.files[i]].set_xlim([-700, 700])
            self.plt[self.files[i]].set_ylim([-700, 700])
            self.plt[self.files[i]].set_zlim([-1000, 700])
            #self.plt[self.files[i]].set_xlabel('x')
            #self.plt[self.files[i]].set_ylabel('y')
            #self.plt[self.files[i]].set_zlabel('z')
            self.plt[self.files[i]].scats = []
            self.plt[self.files[i]].lns = []
            self.plt[self.files[i]].axes.view_init(azim=60, elev=10)
            self.plt[self.files[i]].axes.set_xticklabels([])
            self.plt[self.files[i]].axes.set_yticklabels([])
            self.plt[self.files[i]].axes.set_zticklabels([])
            #self.plt[self.files[i]].axis('off')
            plt.title(titles[i], fontsize=16)

        self.linecolors = ['black', '#0780ea', '#5e01a0', '#2bba00', '#e08302', '#f94e3e']

    def update(self, frame):
        for i in range(len(self.files)):
            for scat in self.plt[self.files[i]].scats:
                scat.remove()
            for ln in self.plt[self.files[i]].lns:
                self.plt[self.files[i]].lines.pop(0)
            self.plt[self.files[i]].scats = []
            self.plt[self.files[i]].lns = []

            for j in range(len(self.chain)):
                self.plt[self.files[i]].lns.append(self.plt[self.files[i]].plot3D(self.data[self.files[i]][frame, self.chain[j], 0], self.data[self.files[i]][frame, self.chain[j], 1], self.data[self.files[i]][frame, self.chain[j], 2], linewidth=2.0, color=self.linecolors[i]))

    def plot(self):
        ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40, repeat=False)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        ani.save('Human.mp4', writer='ffmpeg', fps=25)
        plt.show()


plot = plot_human(data)
plot.plot()
