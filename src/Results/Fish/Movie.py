from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy.io as sio


files = ['gt', 'hmr', 'erd', 'resgru', ]
titles = ['Ground Truth', 'AHMR (Ours)', 'ERD', 'Res-GRU', ]

data = {}
for file in files:
    datafile = sio.loadmat('./' + file + '.mat')
    data[file] = datafile[list(datafile.keys())[3]]

class plot_fish(object):

    def __init__(self, data):

        self.data = data
        self.files = list(data.keys())
        self.nframes = data[self.files[0]].shape[0]

        self.chain = np.arange(0,21)

        self.scattersize = np.zeros(21)
        for k in range(5):
            self.scattersize[k] = 180 - k * 30
        for k in range(5, 21):
            self.scattersize[k] = 30 - (k - 5) * 2

        self.fig = plt.figure()
        self.fig.set_size_inches(16, 9, True)

        self.plt = {}
        plt.title("Forecasting on Zebrafish\n", fontsize=18, fontweight='bold')
        plt.axis('off')
        for i in range(len(self.files)):
            self.plt[self.files[i]] = self.fig.add_subplot(2, 2, i+1, projection='3d')
            self.plt[self.files[i]].set_xlim([-2000, 2000])
            self.plt[self.files[i]].set_ylim([-2000, 2000])
            self.plt[self.files[i]].set_zlim([-1500, 1500])
            #self.plt[self.files[i]].set_xlabel('x')
            #self.plt[self.files[i]].set_ylabel('y')
            #self.plt[self.files[i]].set_zlabel('z')
            self.plt[self.files[i]].scats = []
            self.plt[self.files[i]].lns = []
            self.plt[self.files[i]].axes.view_init(azim=10, elev=60)
            self.plt[self.files[i]].axes.set_xticklabels([])
            self.plt[self.files[i]].axes.set_yticklabels([])
            self.plt[self.files[i]].axes.set_zticklabels([])
            self.plt[self.files[i]].axis('off')
            plt.title(titles[i], fontsize=16)

        self.linecolors = ['black', '#f94e3e', '#0780ea', '#2bba00']

    def update(self, frame):
        for i in range(len(self.files)):
            for scat in self.plt[self.files[i]].scats:
                scat.remove()
            for ln in self.plt[self.files[i]].lns:
                self.plt[self.files[i]].lines.pop(0)
            self.plt[self.files[i]].scats = []
            self.plt[self.files[i]].lns = []

            self.plt[self.files[i]].lns.append(self.plt[self.files[i]].plot3D(self.data[self.files[i]][frame, self.chain, 0], self.data[self.files[i]][frame, self.chain, 1], self.data[self.files[i]][frame, self.chain, 2], linewidth=2.0, color=self.linecolors[i]))
            for k in range(21):
                self.plt[self.files[i]].scats.append(self.plt[self.files[i]].axes.scatter3D(self.data[self.files[i]][frame, self.chain[k], 0], self.data[self.files[i]][frame, self.chain[k], 1], self.data[self.files[i]][frame, self.chain[k], 2], color=self.linecolors[i], s=self.scattersize[k], alpha=1.0 - k * 0.03))

    def plot(self):
        ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40, repeat=False)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.rcParams['animation.ffmpeg_path'] = '../ffmpeg.exe'
        FFwriter = animation.FFMpegWriter(fps=10)
        ani.save('Fish.mp4', writer=FFwriter, dpi=300)
        plt.show()


plot = plot_fish(data)
plot.plot()
