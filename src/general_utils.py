from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import sys
import numpy as np
import copy


def rotmat2expmap(R):
    theta = np.arccos((np.trace(R) - 1) / 2.0)
    if theta < 1e-6:
        A = np.zeros((3, 1))
    else:
        A = theta / (2 * np.sin(theta)) * np.array([[R[2, 1] - R[1, 2]], [R[0, 2] - R[2, 0]], [R[1, 0] - R[0, 1]]])

    return A


def expmap2rotmat(A):
    theta = np.linalg.norm(A)
    if theta == 0:
        R = np.identity(3)
    else:
        A = A / theta
        cross_matrix = np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])
        R = np.identity(3) + np.sin(theta) * cross_matrix + (1 - np.cos(theta)) * np.matmul(cross_matrix, cross_matrix)

    return R


def lietomatrix(angle, trans):
    R = expmap2rotmat(angle)
    T = trans
    SEmatrix = np.concatenate((np.concatenate((R, T.reshape(3, 1)), axis=1), np.array([[0, 0, 0, 1]])))

    return SEmatrix


def findrot(u, v):
    w = np.cross(u, v)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-6:
        A = np.zeros(3)
    else:
        A = w / w_norm * np.arccos(np.dot(u, v))

    return A


def computelie(lie_params):
    njoints = np.shape(lie_params)[0]
    A = np.zeros((njoints, 4, 4))

    for j in range(njoints):
        if j == 0:
            A[j, :, :] = lietomatrix(lie_params[j, 0:3].T, lie_params[j, 3:6].T)
        else:
            A[j, :, :] = np.matmul(np.squeeze(A[j - 1, :, :]),
                                   lietomatrix(lie_params[j, 0:3].T, lie_params[j, 3:6].T))

    joint_xyz = np.zeros((njoints, 3))

    for j in range(njoints):
        coor = np.array([0, 0, 0, 1]).reshape((4, 1))
        xyz = np.matmul(np.squeeze(A[j, :, :]), coor)
        joint_xyz[j, :] = xyz[0:3, 0]

    return joint_xyz


def rotmat2euler(R):
    if R[0, 2] == 1 or R[0, 2] == -1:
        E3 = 0
        dlta = np.arctan2(R[0, 1], R[0, 2])
        if R[0, 2] == -1:
            E2 = np.pi / 2
            E1 = E3 + dlta
        else:
            E2 = -np.pi / 2
            E1 = -E3 + dlta
    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))

    eul = np.array([E1, E2, E3])

    return eul


def euler2rotmat(angle):
    a = angle[0]
    b = angle[1]
    c = angle[2]

    a1 = np.array([1, 0, 0])
    a2 = np.array([0, np.cos(a), -np.sin(a)])
    a3 = np.array([0, np.sin(a), np.cos(a)])

    A = np.array([a1, a2, a3])

    b1 = np.array([np.cos(b), 0, np.sin(b)])
    b2 = np.array([0, 1, 0])
    b3 = np.array([-np.sin(b), 0, np.cos(b)])

    B = np.array([b1, b2, b3])

    c1 = np.array([np.cos(c), -np.sin(c), 0])
    c2 = np.array([np.sin(c), np.cos(c), 0])
    c3 = np.array([0, 0, 1])

    C = np.array([c1, c2, c3])

    R = np.matmul(np.matmul(A, B), C)

    return R


def eulertomatrix(angle, trans):
    R = euler2rotmat(angle)
    T = trans
    SEmatrix = np.concatenate((np.concatenate((R, T.reshape(3, 1)), axis=1), np.array([[0, 0, 0, 1]])))

    return SEmatrix


def forward_kinematics(data, config):
    bone = config.bone

    nframes = data.shape[0]
    data = data.reshape([nframes, -1, 3])

    njoints = data.shape[1] + 1

    lie_params = np.zeros([nframes, njoints, 6])

    for i in range(njoints - 1):
        lie_params[:, i, 0:3] = data[:, i, :]

    lie_params[:, :, 3:6] = bone
    lie_params[:, 0, 3:6] = np.zeros([3])

    joint_xyz_f = np.zeros([nframes, njoints, 3])

    for i in range(nframes):
        joint_xyz_f[i, :, :] = computelie(np.squeeze(lie_params[i, :, :]))
    return joint_xyz_f


def inverse_kinematics(data, config):
    nframes = data.shape[0]
    joint_xyz = data.reshape([nframes, -1, 3])
    njoints = joint_xyz.shape[1]

    lie_parameters = np.zeros((nframes, njoints*3+3))

    for i in range(nframes):
        for j in range(njoints - 2, -1, -1):
            v = np.squeeze(joint_xyz[i, j + 1, :] - joint_xyz[i, j, :])
            vhat = v / np.linalg.norm(v)

            if j == 0:
                uhat = np.array([1, 0, 0])
            else:
                u = np.squeeze(joint_xyz[i, j, :] - joint_xyz[i, j - 1, :])
                uhat = u / np.linalg.norm(u)

            if np.linalg.norm(u) == 0:
                uhat = np.array([1, 0, 0])
            if np.linalg.norm(v) == 0:
                vhat = np.array([1, 0, 0])

            A = expmap2rotmat(findrot(np.array([1, 0, 0]), np.squeeze(uhat))).T
            B = expmap2rotmat(findrot(np.array([1, 0, 0]), np.squeeze(vhat)))
            lie_parameters[i, (j+1)*3:(j+2)*3] = np.squeeze(rotmat2expmap(np.matmul(A, B)))

    return lie_parameters


def normalize_data(data, data_mean, data_std, dim_to_use):
    """
    Copied from https://github.com/una-dinosauria/human-motion-prediction
    """
    data_out = {}

    for key in data.keys():
        data_out[key] = np.divide((data[key] - data_mean), data_std)
        data_out[key] = data_out[key][:, dim_to_use]

    return data_out


def normalization_stats(completeData):
    """
    Copied from https://github.com/una-dinosauria/human-motion-prediction
    """
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return [data_mean, data_std, dimensions_to_ignore, dimensions_to_use]


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore):
    """
    Copied from https://github.com/una-dinosauria/human-motion-prediction
    """

    T = normalizedData.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    origData[:, dimensions_to_use] = normalizedData

    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    origData = np.multiply(origData, stdMat) + meanMat
    return origData


class Progbar(object):
    """Progbar class copied from https://github.com/fchollet/keras/

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


def get_batch(config, data):
    # Generate x_train and y_train from train_set

    # Select entries at random
    all_keys = list(data.keys())
    batch_size = config.batch_size
    chosen_keys = np.random.choice(len(all_keys), batch_size)

    total_frames = config.input_window_size + config.output_window_size

    encoder_inputs = np.zeros((batch_size, config.input_window_size - 1, config.input_size), dtype=float)
    decoder_inputs = np.zeros((batch_size, config.output_window_size, config.input_size), dtype=float)
    decoder_outputs = np.zeros((batch_size, config.output_window_size, config.input_size), dtype=float)

    for i in range(batch_size):
        the_key = all_keys[chosen_keys[i]]

        n, _ = data[the_key].shape

        # Random sample
        idx = np.random.randint(16, n - total_frames)

        # Select the window around the sampled index
        data_sel = data[the_key][idx:idx + total_frames, :]

        # Organize into batch_size x window_size x input_size
        encoder_inputs[i, :, 0:config.input_size] = data_sel[0:config.input_window_size - 1, :]  # x_train
        decoder_inputs[i, :, 0:config.input_size] = data_sel[
                                                    config.input_window_size - 1:config.input_window_size + config.output_window_size - 1,
                                                    :]  # decoder_in_train
        decoder_outputs[i, :, 0:config.input_size] = data_sel[config.input_window_size:, 0:config.input_size]  # y_train

    return encoder_inputs, decoder_inputs, decoder_outputs


def create_directory(config):
    # Checkpoint directory

    folder_dir = config.dataset + '/' + config.datatype + '_' + config.loss + '_' + config.model
    if config.model == 'HMR':
        folder_dir += '_RecurrentSteps=' + str(config.recurrent_steps) + '_' + 'ContextWindow=' + str(
            config.context_window) + '_' + 'hiddenSize=' + str(config.hidden_size)

    folder_dir += '/' + config.filename + '/'
    folder_dir += 'inputWindow=' + str(config.input_window_size) + '_outputWindow=' + str(
        config.output_window_size) + '/'

    checkpoint_dir = './checkpoint/' + folder_dir
    output_dir = './output/' + folder_dir

    return [checkpoint_dir, output_dir]


def forward_kinematics_h36m(angles, config):
    nframes = len(angles)
    njoints = len(config.parents)
    xyz = np.zeros([nframes, njoints, 3])
    R = np.zeros([njoints, 3, 3])

    angles = np.reshape(angles, [nframes, -1, 3])
    if angles.shape[1] > njoints:
        angles = angles[:, 1:, :]

    for frame in range(nframes):
        for i in range(njoints):
            if config.parents[i] == -1:  # Root node
                R[i] = np.eye(3)
            else:
                xyz[frame, i] = config.offsets[i].dot(R[config.parents[i]]) + xyz[frame, config.parents[i]]
                R[i] = expmap2rotmat(angles[frame, i]).dot(R[config.parents[i]])
    return xyz[:,:,[0,2,1]]


def fk(data, config):
    if config.dataset == 'Human':
        xyz = forward_kinematics_h36m(data, config)
    else:
        xyz = forward_kinematics(data, config)
    return xyz


def mean_euler_error(config, action, y_predict, y_test):
    # Convert from exponential map to Euler angles
    n_batch = y_predict.shape[0]
    nframes = y_predict.shape[1]

    mean_errors = np.zeros([n_batch, nframes])
    for i in range(n_batch):
        for j in range(nframes):
            pred = copy.deepcopy(y_predict[i])
            gt = copy.deepcopy(y_test[i])
            for k in np.arange(3, pred.shape[1] - 2, 3):
                pred[j, k:k + 3] = rotmat2euler(expmap2rotmat(pred[j, k:k + 3]))
                gt[j, k:k + 3] = rotmat2euler(expmap2rotmat(gt[j, k:k + 3]))
        pred[:, 0:6] = 0
        gt[:, 0:6] = 0

        idx_to_use = np.where(np.std(gt, 0) > 1e-4)[0]

        euc_error = np.power(gt[:, idx_to_use] - pred[:, idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt(euc_error)
        mean_errors[i, :] = euc_error

    mme = np.mean(mean_errors, 0)

    toprint_idx = np.array([1, 3, 7, 9, 13, 15, 17, 24])
    idx = np.where(toprint_idx < len(mme))[0]
    print("\n" + action)
    toprint_list = ["& {:.2f} ".format(mme[toprint_idx[i]]) for i in idx]
    print("".join(toprint_list))

    mme_mean = np.mean(mme[toprint_idx[idx]])
    return mme, mme_mean
