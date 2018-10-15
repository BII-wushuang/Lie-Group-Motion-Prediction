import numpy as np
import tensorflow as tf
import scipy.io
import general_utils as data_utils


def read_data(config, training=False):
    if config.dataset == 'Human':
        if config.datatype == 'xyz':
            train_set, test_set, x_test, y_test, dec_in_test, config = read_human(config, training)
        else:
            train_set, test_set, x_test, y_test, dec_in_test, config = read_h36m(config, training)
    elif config.dataset == 'Mouse':
        train_set, test_set, x_test, y_test, dec_in_test, config = read_mouse(config, training)
    elif config.dataset == 'Fish':
        train_set, test_set, x_test, y_test, dec_in_test, config = read_fish(config, training)

    return [train_set, test_set, x_test, y_test, dec_in_test, config]


def read_human(config, training):
    if config.datatype == 'lie':
        train_path = './data/Human/Train/train_lie/'
        x_test_path = './data/Human/Test/x_test_lie/'
        y_test_path = './data/Human/Test/y_test_lie/'
    elif config.datatype == 'xyz':
        train_path = './data/Human/Train/train_xyz/'
        x_test_path = './data/Human/Test/x_test_xyz/'
        y_test_path = './data/Human/Test/y_test_xyz/'
    if config.filename == 'all':
        actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting',
               'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']
    else:
        actions = [config.filename]
    train_subjects = ['S1', 'S6', 'S7', 'S8', 'S9', 'S11']
    test_subjects = ['S5']

    seq_length_in = config.input_window_size

    if training:
        seq_length_out = config.output_window_size
    else:
        seq_length_out = config.test_output_window

    # Kinematic chain configuration
    skip = config.skip
    bone_skip = skip[0:-1]

    # Read a base file to establish kinematic chain configurations
    if config.datatype == 'lie':
        filename = y_test_path + actions[0] + '_1_lie.mat'
        rawdata = scipy.io.loadmat(filename)
        matdict_key = list(rawdata.keys())[3]
        rawdata = rawdata[matdict_key]
        njoints = rawdata.shape[1]
        bone = np.zeros([njoints, 3])

        # Bone lengths
        for i in range(njoints):
            if i in bone_skip:
                continue
            else:
                bone[i, 0] = round(rawdata[0, i, 3], 2)
    elif config.datatype == 'xyz':
        filename = y_test_path + actions[0] + '_1_xyz.mat'
        rawdata = scipy.io.loadmat(filename)
        matdict_key = list(rawdata.keys())[3]
        rawdata = rawdata[matdict_key]
        njoints = rawdata.shape[1]
        bone = np.zeros([njoints, 3])

        # Bone lengths
        for i in range(njoints):
            if i in bone_skip:
                continue
            else:
                bone[i, 0] = round(np.linalg.norm(rawdata[0, i, :] - rawdata[0, i - 1, :]), 2)

    bone_params = tf.convert_to_tensor(bone)
    bone_params = tf.cast(bone_params, tf.float32)

    config.bone = bone
    config.bone_params = bone_params

    config.output_window_size = np.min([config.output_window_size, rawdata.shape[0]])
    config.test_output_window = np.min([config.test_output_window, rawdata.shape[0]])
    seq_length_out = np.min([seq_length_out, rawdata.shape[0]])

    if training:
        print("Reading {0} data for training: Input Sequence Length = {1}, Output Sequence Length = {2}.".format(config.dataset, seq_length_in, seq_length_out))
    else:
        print("Reading {0} data for testing: Input Sequence Length = {1}, Output Sequence Length = {2}.".format(config.dataset, seq_length_in, seq_length_out))

    # Read and prepare training data
    train_set = {}
    for action in actions:
        for id in train_subjects:
            for i in range(2):

                if config.datatype == 'lie':
                    filename = train_path + id + '_' + action + '_' + str(i+1) + '_lie.mat'
                    rawdata = scipy.io.loadmat(filename)
                    matdict_key = list(rawdata.keys())[3]
                    rawdata = rawdata[matdict_key]

                    nframes = rawdata.shape[0]
                    data = np.zeros([nframes, njoints + 1, 3])
                    # Reorganising the Lie algebra parameters to remove redundancy
                    data[:, 0, :] = rawdata[:, 0, 3:6]
                    data[:, 1:, :] = rawdata[:, :, 0:3]
                    data = np.delete(data, [skip], axis=1)
                    data = np.around(data, 5)
                    data = data.reshape(data.shape[0], -1)
                    train_set[action + '_' + id + '_'+ str(i+1)] = data

                elif config.datatype == 'xyz':
                    filename = train_path + id + '_' + action + '_' + str(i+1) + '_xyz.mat'
                    rawdata = scipy.io.loadmat(filename)
                    matdict_key = list(rawdata.keys())[3]
                    rawdata = rawdata[matdict_key]
                    data = rawdata.reshape(rawdata.shape[0], -1)
                    train_set[action + '_' + id + '_'+ str(i+1)] = data

    test_set = {}
    for action in actions:
        for id in test_subjects:
            for i in range(2):

                if config.datatype == 'lie':
                    filename = train_path + id + '_' + action + '_' + str(i + 1) + '_lie.mat'
                    rawdata = scipy.io.loadmat(filename)
                    matdict_key = list(rawdata.keys())[3]
                    rawdata = rawdata[matdict_key]

                    nframes = rawdata.shape[0]
                    data = np.zeros([nframes, njoints + 1, 3])
                    # Reorganising the Lie algebra parameters to remove redundancy
                    data[:, 0, :] = rawdata[:, 0, 3:6]
                    data[:, 1:, :] = rawdata[:, :, 0:3]
                    data = np.delete(data, [skip], axis=1)
                    data = np.around(data, 5)
                    data = data.reshape(data.shape[0], -1)
                    test_set[action + '_' + id + '_' + str(i + 1)] = data

                elif config.datatype == 'xyz':
                    filename = train_path + id + '_' + action + '_' + str(i + 1) + '_xyz.mat'
                    rawdata = scipy.io.loadmat(filename)
                    matdict_key = list(rawdata.keys())[3]
                    rawdata = rawdata[matdict_key]
                    data = rawdata.reshape(rawdata.shape[0], -1)
                    test_set[action + '_' + id + '_' + str(i + 1)] = data

    # Read and prepare test data
    x_test_dict = {}
    y_test_dict = {}
    dec_in_test_dict = {}

    for action in actions:
        x_test = []
        y_test = []

        for i in range(8):
            if config.datatype == 'lie':
                x_filename = x_test_path + action + '_' + str(i) + '_lie.mat'
                y_filename = y_test_path + action + '_' + str(i) + '_lie.mat'

                x_rawdata = scipy.io.loadmat(x_filename)
                matdict_key = list(x_rawdata.keys())[3]
                x_rawdata = x_rawdata[matdict_key]

                y_rawdata = scipy.io.loadmat(y_filename)
                matdict_key = list(y_rawdata.keys())[3]
                y_rawdata = y_rawdata[matdict_key]

                x_data = np.zeros([config.input_window_size, njoints + 1, 3])
                # Reorganising the Lie algebra parameters to remove redundancy
                x_data[:, 0, :] = x_rawdata[:, 0, 3:6]
                x_data[:, 1:, :] = x_rawdata[:, :, 0:3]
                x_data = np.delete(x_data, [skip], axis=1)
                x_data = np.around(x_data, 5)
                x_data = x_data.reshape(x_data.shape[0], -1)
                x_test.append(x_data)

                y_data = np.zeros([seq_length_out, njoints + 1, 3])
                # Reorganising the Lie algebra parameters to remove redundancy
                y_data[:, 0, :] = y_rawdata[:seq_length_out, 0, 3:6]
                y_data[:, 1:, :] = y_rawdata[:seq_length_out, :, 0:3]
                y_data = np.delete(y_data, [skip], axis=1)
                y_data = np.around(y_data, 5)
                y_data = y_data.reshape(y_data.shape[0], -1)

                y_test.append(y_data)

            elif config.datatype == 'xyz':
                x_filename = x_test_path + action + '_' + str(i) + '_xyz.mat'
                y_filename = y_test_path + action + '_' + str(i) + '_xyz.mat'

                x_rawdata = scipy.io.loadmat(x_filename)
                matdict_key = list(x_rawdata.keys())[3]
                x_rawdata = x_rawdata[matdict_key]

                y_rawdata = scipy.io.loadmat(y_filename)
                matdict_key = list(y_rawdata.keys())[3]
                y_rawdata = y_rawdata[matdict_key]

                x_data = x_rawdata.reshape(x_rawdata.shape[0], -1)
                x_test.append(x_data)

                y_data = y_rawdata.reshape(y_rawdata.shape[0], -1)
                y_data = y_data[:seq_length_out, :]
                y_test.append(y_data)

        x_test = np.array(x_test)
        y_test = np.array(y_test)
        dec_in_test = np.concatenate((np.reshape(x_test[:, -1, :], [x_test.shape[0], 1, x_test.shape[2]]), y_test[:, 0:-1, :]), axis=1)
        x_test = x_test[:, 0:-1, :]

        x_test_dict[action] = x_test
        y_test_dict[action] = y_test
        dec_in_test_dict[action] = dec_in_test

    config.input_size = 81

    print("Done reading data.")

    return [train_set, test_set, x_test_dict, y_test_dict, dec_in_test_dict, config]


def read_h36m(config, training):
    seq_length_in = config.input_window_size
    if training:
        seq_length_out = config.output_window_size
    else:
        seq_length_out = config.test_output_window

    if training:
        print("Reading {0} data for training: Input Sequence Length = {1}, Output Sequence Length = {2}.".format(config.dataset, seq_length_in, seq_length_out))
    else:
        print("Reading {0} data for testing: Input Sequence Length = {1}, Output Sequence Length = {2}.".format(config.dataset, seq_length_in, seq_length_out))

    if config.filename == 'all':
        actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting',
                   'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']
    else:
        actions = [config.filename]

    train_subjects = [1,6,7,8,9,11]
    test_subjects = [5]

    data_dir = './data/h3.6m/dataset'
    train_set, complete_train = data_utils.load_data(data_dir, train_subjects, actions)
    test_set, complete_test = data_utils.load_data(data_dir, test_subjects, actions)

    # Compute normalization stats
    data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)
    config.data_mean = data_mean
    config.data_std = data_std
    config.dim_to_ignore = dim_to_ignore
    config.dim_to_use = dim_to_use

    # Normalize: subtract mean, divide by std
    train_set = data_utils.normalize_data(train_set, data_mean, data_std, dim_to_use)
    test_set = data_utils.normalize_data(test_set, data_mean, data_std, dim_to_use)

    config.input_size = train_set[list(train_set.keys())[0]].shape[1]

    x_test = {}
    y_test = {}
    dec_in_test = {}
    for action in actions:
        encoder_inputs, decoder_inputs, decoder_outputs = get_batch_srnn(config, test_set, action, seq_length_out)
        x_test[action] = encoder_inputs
        y_test[action] = decoder_outputs
        dec_in_test[action] = decoder_inputs

    print("Done reading data.")
    config.chain_idx = [np.array([0, 1, 2, 3, 4, 5]),
                        np.array([0, 6, 7, 8, 9, 10]),
                        np.array([0, 12, 13, 14, 15]),
                        np.array([13, 17, 18, 19, 22, 19, 21]),
                        np.array([13, 25, 26, 27, 30, 27, 29])]

    return [train_set, test_set, x_test, y_test, dec_in_test, config]


def read_general(config, training, train_subjects, test_subjects, train_path, x_test_path, y_test_path):

    seq_length_in = config.input_window_size

    if training:
        seq_length_out = config.output_window_size
    else:
        seq_length_out = config.test_output_window

    # Kinematic chain configuration
    skip = config.skip
    bone_skip = skip[0:-1]

    # Read a base file to establish kinematic chain configurations
    if config.datatype == 'lie':
        filename = y_test_path + 'test_0' + '_lie.mat'
        rawdata = scipy.io.loadmat(filename)
        matdict_key = list(rawdata.keys())[3]
        rawdata = rawdata[matdict_key]
        njoints = rawdata.shape[1]
        bone = np.zeros([njoints, 3])

        # Bone lengths
        for i in range(njoints):
            if i in bone_skip:
                continue
            else:
                bone[i, 0] = round(rawdata[0, i, 3], 2)
    elif config.datatype == 'xyz':
        filename = y_test_path + 'test_0' + '.mat'
        rawdata = scipy.io.loadmat(filename)
        matdict_key = list(rawdata.keys())[3]
        rawdata = rawdata[matdict_key]
        njoints = rawdata.shape[1]
        bone = np.zeros([njoints, 3])

        # Bone lengths
        for i in range(njoints):
            if i in bone_skip:
                continue
            else:
                bone[i, 0] = round(np.linalg.norm(rawdata[0, i, :] - rawdata[0, i - 1, :]), 2)

    bone_params = tf.convert_to_tensor(bone)
    bone_params = tf.cast(bone_params, tf.float32)

    config.bone = bone
    config.bone_params = bone_params

    config.output_window_size = np.min([config.output_window_size, rawdata.shape[0]])
    config.test_output_window = np.min([config.test_output_window, rawdata.shape[0]])
    seq_length_out = np.min([seq_length_out, rawdata.shape[0]])

    if training:
        print("Reading {0} data for training: Input Sequence Length = {1}, Output Sequence Length = {2}.".format(config.dataset, seq_length_in, seq_length_out))
    else:
        print("Reading {0} data for testing: Input Sequence Length = {1}, Output Sequence Length = {2}.".format(config.dataset, seq_length_in, seq_length_out))

    # Read and prepare training data
    train_set = {}

    for id in train_subjects:

        if config.datatype == 'lie':
            filename = train_path + id + '_lie.mat'
            rawdata = scipy.io.loadmat(filename)
            matdict_key = list(rawdata.keys())[3]
            rawdata = rawdata[matdict_key]

            nframes = rawdata.shape[0]
            data = np.zeros([nframes, njoints + 1, 3])
            # Reorganising the Lie algebra parameters to remove redundancy
            data[:, 0, :] = rawdata[:, 0, 3:6]
            data[:, 1:, :] = rawdata[:, :, 0:3]
            data = np.delete(data, [skip], axis=1)
            data = np.around(data, 5)
            data = data.reshape(data.shape[0], -1)
            train_set[id] = data

        elif config.datatype == 'xyz':
            filename = train_path + id + '_xyz.mat'
            rawdata = scipy.io.loadmat(filename)
            matdict_key = list(rawdata.keys())[3]
            rawdata = rawdata[matdict_key]
            data = rawdata.reshape(rawdata.shape[0], -1)
            train_set[id] = data

    test_set = {}

    for id in test_subjects:

        if config.datatype == 'lie':
            filename = train_path + id + '_lie.mat'
            rawdata = scipy.io.loadmat(filename)
            matdict_key = list(rawdata.keys())[3]
            rawdata = rawdata[matdict_key]

            nframes = rawdata.shape[0]
            data = np.zeros([nframes, njoints + 1, 3])
            # Reorganising the Lie algebra parameters to remove redundancy
            data[:, 0, :] = rawdata[:, 0, 3:6]
            data[:, 1:, :] = rawdata[:, :, 0:3]
            data = np.delete(data, [skip], axis=1)
            data = np.around(data, 5)
            data = data.reshape(data.shape[0], -1)
            test_set[id] = data

        elif config.datatype == 'xyz':
            filename = train_path + id + '_xyz.mat'
            rawdata = scipy.io.loadmat(filename)
            matdict_key = list(rawdata.keys())[3]
            rawdata = rawdata[matdict_key]
            data = rawdata.reshape(rawdata.shape[0], -1)
            test_set[id] = data

    # Read and prepare test data
    x_test = []
    y_test = []

    for i in range(8):

        if config.datatype == 'lie':
            x_filename = x_test_path + 'test_' + str(i) + '_lie.mat'
            y_filename = y_test_path + 'test_' + str(i) + '_lie.mat'

            x_rawdata = scipy.io.loadmat(x_filename)
            matdict_key = list(x_rawdata.keys())[3]
            x_rawdata = x_rawdata[matdict_key]

            y_rawdata = scipy.io.loadmat(y_filename)
            matdict_key = list(y_rawdata.keys())[3]
            y_rawdata = y_rawdata[matdict_key]

            x_data = np.zeros([seq_length_in, njoints + 1, 3])
            # Reorganising the Lie algebra parameters to remove redundancy
            x_data[:, 0, :] = x_rawdata[:, 0, 3:6]
            x_data[:, 1:, :] = x_rawdata[:, :, 0:3]
            x_data = np.delete(x_data, [skip], axis=1)
            x_data = np.around(x_data, 5)
            x_data = x_data.reshape(x_data.shape[0], -1)
            x_test.append(x_data)

            y_data = np.zeros([seq_length_out, njoints + 1, 3])
            # Reorganising the Lie algebra parameters to remove redundancy
            y_data[:, 0, :] = y_rawdata[:seq_length_out, 0, 3:6]
            y_data[:, 1:, :] = y_rawdata[:seq_length_out, :, 0:3]
            y_data = np.delete(y_data, [skip], axis=1)
            y_data = np.around(y_data, 5)
            y_data = y_data.reshape(y_data.shape[0], -1)
            y_test.append(y_data)

        elif config.datatype == 'xyz':
            x_filename = x_test_path + 'test_' + str(i) + '.mat'
            y_filename = y_test_path + 'test_' + str(i) + '.mat'

            x_rawdata = scipy.io.loadmat(x_filename)
            matdict_key = list(x_rawdata.keys())[3]
            x_rawdata = x_rawdata[matdict_key]

            y_rawdata = scipy.io.loadmat(y_filename)
            matdict_key = list(y_rawdata.keys())[3]
            y_rawdata = y_rawdata[matdict_key]

            x_data = x_rawdata.reshape(x_rawdata.shape[0], -1)
            x_test.append(x_data)

            y_data = y_rawdata.reshape(y_rawdata.shape[0], -1)
            y_data = y_data[:seq_length_out, :]
            y_test.append(y_data)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    dec_in_test = np.concatenate(
        (np.reshape(x_test[:, -1, :], [x_test.shape[0], 1, x_test.shape[2]]), y_test[:, 0:-1, :]), axis=1)
    x_test = x_test[:, 0:-1, :]

    x_test_dict = {}
    y_test_dict = {}
    dec_in_test_dict = {}

    x_test_dict['default'] = x_test
    y_test_dict['default'] = y_test
    dec_in_test_dict['default'] = dec_in_test

    print("Done reading data.")

    config.input_size = x_test.shape[2]

    return [train_set, test_set, x_test_dict, y_test_dict, dec_in_test_dict, config]


def read_mouse(config, training):
    if config.datatype == 'lie':
        train_path = './data/Mouse/Train/train_lie/'
        x_test_path = './data/Mouse/Test/x_test_lie/'
        y_test_path = './data/Mouse/Test/y_test_lie/'
    elif config.datatype == 'xyz':
        train_path = './data/Mouse/Train/train_xyz/'
        x_test_path = './data/Mouse/Test/x_test_xyz/'
        y_test_path = './data/Mouse/Test/y_test_xyz/'

    train_subjects = ['S1', 'S3', 'S4']
    test_subjects = ['S2']

    seq_length_in = config.input_window_size

    if training:
        seq_length_out = config.output_window_size
    else:
        seq_length_out = config.test_output_window

    filename = y_test_path + 'test_0' + '_lie.mat'
    rawdata = scipy.io.loadmat(filename)
    matdict_key = list(rawdata.keys())[3]
    rawdata = rawdata[matdict_key]

    config.output_window_size = np.min([config.output_window_size, rawdata.shape[0]])
    config.test_output_window = np.min([config.test_output_window, rawdata.shape[0]])
    seq_length_out = np.min([seq_length_out, rawdata.shape[0]])

    if training:
        print("Reading {0} data for training: Input Sequence Length = {1}, Output Sequence Length = {2}.".format(
            config.dataset, seq_length_in, seq_length_out))
    else:
        print("Reading {0} data for testing: Input Sequence Length = {1}, Output Sequence Length = {2}.".format(
            config.dataset, seq_length_in, seq_length_out))

    # Read and prepare training data
    train_set = {}

    for id in train_subjects:

        if config.datatype == 'lie':
            filename = train_path + id + '_lie.mat'
            rawdata = scipy.io.loadmat(filename)
            matdict_key = list(rawdata.keys())[3]
            rawdata = rawdata[matdict_key]

            nframes = rawdata.shape[0]
            train_set[id] = rawdata

        elif config.datatype == 'xyz':
            filename = train_path + id + '_xyz.mat'
            rawdata = scipy.io.loadmat(filename)
            matdict_key = list(rawdata.keys())[3]
            rawdata = rawdata[matdict_key]
            data = rawdata.reshape(rawdata.shape[0], -1)
            train_set[id] = data

    test_set = {}

    for id in test_subjects:

        if config.datatype == 'lie':
            filename = train_path + id + '_lie.mat'
            rawdata = scipy.io.loadmat(filename)
            matdict_key = list(rawdata.keys())[3]
            rawdata = rawdata[matdict_key]

            nframes = rawdata.shape[0]
            test_set[id] = rawdata

        elif config.datatype == 'xyz':
            filename = train_path + id + '_xyz.mat'
            rawdata = scipy.io.loadmat(filename)
            matdict_key = list(rawdata.keys())[3]
            rawdata = rawdata[matdict_key]
            data = rawdata.reshape(rawdata.shape[0], -1)
            test_set[id] = data

    # Read and prepare test data
    x_test = []
    y_test = []

    for i in range(8):

        if config.datatype == 'lie':
            x_filename = x_test_path + 'test_' + str(i) + '_lie.mat'
            y_filename = y_test_path + 'test_' + str(i) + '_lie.mat'

            x_rawdata = scipy.io.loadmat(x_filename)
            matdict_key = list(x_rawdata.keys())[3]
            x_rawdata = x_rawdata[matdict_key]

            y_rawdata = scipy.io.loadmat(y_filename)
            matdict_key = list(y_rawdata.keys())[3]
            y_rawdata = y_rawdata[matdict_key]

            x_test.append(x_rawdata)

            y_test.append(y_rawdata)

        elif config.datatype == 'xyz':
            x_filename = x_test_path + 'test_' + str(i) + '.mat'
            y_filename = y_test_path + 'test_' + str(i) + '.mat'

            x_rawdata = scipy.io.loadmat(x_filename)
            matdict_key = list(x_rawdata.keys())[3]
            x_rawdata = x_rawdata[matdict_key]

            y_rawdata = scipy.io.loadmat(y_filename)
            matdict_key = list(y_rawdata.keys())[3]
            y_rawdata = y_rawdata[matdict_key]

            x_data = x_rawdata.reshape(x_rawdata.shape[0], -1)
            x_test.append(x_data)

            y_data = y_rawdata.reshape(y_rawdata.shape[0], -1)
            y_data = y_data[:seq_length_out, :]
            y_test.append(y_data)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    dec_in_test = np.concatenate(
        (np.reshape(x_test[:, -1, :], [x_test.shape[0], 1, x_test.shape[2]]), y_test[:, 0:-1, :]), axis=1)
    x_test = x_test[:, 0:-1, :]

    x_test_dict = {}
    y_test_dict = {}
    dec_in_test_dict = {}

    x_test_dict['default'] = x_test
    y_test_dict['default'] = y_test
    dec_in_test_dict['default'] = dec_in_test

    print("Done reading data.")

    config.input_size = x_test.shape[2]

    return [train_set, test_set, x_test_dict, y_test_dict, dec_in_test_dict, config]


def read_mouse2(config, training):

    if config.datatype == 'lie':
        train_path = './data/Mouse/Train/train_lie/'
        x_test_path = './data/Mouse/Test/x_test_lie/'
        y_test_path = './data/Mouse/Test/y_test_lie/'
    elif config.datatype == 'xyz':
        train_path = './data/Mouse/Train/train_xyz/'
        x_test_path = './data/Mouse/Test/x_test_xyz/'
        y_test_path = './data/Mouse/Test/y_test_xyz/'

    train_subjects = ['S1', 'S3', 'S4']
    test_subjects = ['S2']

    train_set, test_set, x_test_dict, y_test_dict, dec_in_test_dict, config = read_general(config, training, train_subjects, test_subjects, train_path, x_test_path, y_test_path)

    return [train_set, test_set, x_test_dict, y_test_dict, dec_in_test_dict, config]


def read_fish(config, training):

    if config.datatype == 'lie':
        train_path = './data/Fish/Train/train_lie/'
        x_test_path = './data/Fish/Test/x_test_lie/'
        y_test_path = './data/Fish/Test/y_test_lie/'
    elif config.datatype == 'xyz':
        train_path = './data/Fish/Train/train_xyz/'
        x_test_path = './data/Fish/Test/x_test_xyz/'
        y_test_path = './data/Fish/Test/y_test_xyz/'

    train_subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S7', 'S8']
    test_subjects = ['S6']

    train_set, test_set, x_test_dict, y_test_dict, dec_in_test_dict, config = read_general(config, training, train_subjects, test_subjects, train_path, x_test_path, y_test_path)

    return [train_set, test_set, x_test_dict, y_test_dict, dec_in_test_dict, config]


def get_batch_srnn(config, data, action, target_seq_len):
    # Obtain SRNN test sequences using the specified random seeds

    frames = {}
    frames[ action ] = find_indices_srnn( data, action )

    batch_size = 8
    subject    = 5
    source_seq_len = config.input_window_size

    seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

    encoder_inputs  = np.zeros( (batch_size, source_seq_len-1, config.input_size), dtype=float )
    decoder_inputs  = np.zeros( (batch_size, target_seq_len, config.input_size), dtype=float )
    decoder_outputs = np.zeros( (batch_size, target_seq_len, config.input_size), dtype=float )

    for i in range( batch_size ):
      _, subsequence, idx = seeds[i]
      idx = idx + 50

      data_sel = data[ (subject, action, subsequence, 'even') ]

      data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

      encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :] #x_test
      decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :] #decoder_in_test
      decoder_outputs[i, :, :] = data_sel[source_seq_len:, :] #y_test

    return [encoder_inputs, decoder_inputs, decoder_outputs]


def find_indices_srnn(data, action):
    """
    Obtain the same action indices as in SRNN using a fixed random seed
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py
    """

    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    subject = 5
    subaction1 = 1
    subaction2 = 2

    T1 = data[(subject, action, subaction1, 'even')].shape[0]
    T2 = data[(subject, action, subaction2, 'even')].shape[0]
    prefix, suffix = 50, 100

    idx = []
    idx.append(rng.randint(16,T1-prefix-suffix))
    idx.append(rng.randint(16,T2-prefix-suffix))
    idx.append(rng.randint(16,T1-prefix-suffix))
    idx.append(rng.randint(16,T2-prefix-suffix))
    idx.append(rng.randint(16,T1-prefix-suffix))
    idx.append(rng.randint(16,T2-prefix-suffix))
    idx.append(rng.randint(16,T1-prefix-suffix))
    idx.append(rng.randint(16,T2-prefix-suffix))

    return idx
