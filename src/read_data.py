import numpy as np
import tensorflow as tf
import scipy.io as sio
import general_utils as data_utils
import copy


def read_data(config, training=False):
    if config.dataset == 'Human':
        train_set, test_set, x_test, y_test, dec_in_test, config = read_human(config, training)
    elif config.dataset == 'Mouse':
        train_set, test_set, x_test, y_test, dec_in_test, config = read_mouse(config, training)
    elif config.dataset == 'Fish':
        train_set, test_set, x_test, y_test, dec_in_test, config = read_fish(config, training)

    return [train_set, test_set, x_test, y_test, dec_in_test, config]


def read_human(config, training):
    seq_length_in = config.input_window_size
    if training:
        seq_length_out = config.output_window_size
    else:
        seq_length_out = config.test_output_window

    if training:
        print("Reading {0} data for training: Input Sequence Length = {1}, Output Sequence Length = {2}.".format(
            config.dataset, seq_length_in, seq_length_out))
    else:
        print("Reading {0} data for testing: Input Sequence Length = {1}, Output Sequence Length = {2}.".format(
            config.dataset, seq_length_in, seq_length_out))

    if config.filename == 'all':
        actions = ['discussion', 'greeting', 'posing', 'walkingdog', 'directions', 'eating', 'phoning','purchases', 'sitting',
                   'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingtogether']
        # actions = ['walking', 'eating', 'smoking', 'discussion', 'directions', 'greeting', 'phoning', 'posing',
        #            'purchases', 'sitting', 'sittingdown', 'takingphoto', 'waiting', 'walkingdog', 'walkingtogether']
    else:
        actions = [config.filename]

    train_set = {}
    complete_train = []
    for subj in [1, 6, 7, 8, 9, 11]:
        for action in actions:
            for subact in [1, 2]:
                if config.datatype == 'lie':
                    filename = '{0}/S{1}_{2}_{3}_lie.mat'.format('./data/h3.6m/Train/train_lie', subj, action, subact)
                    train_set[(subj, action, subact)] = sio.loadmat(filename)['lie_parameters']

                if config.datatype == 'xyz':
                    filename = '{0}/S{1}_{2}_{3}_xyz.mat'.format('./data/h3.6m/Train/train_xyz', subj, action, subact)
                    train_set[(subj, action, subact)] = sio.loadmat(filename)['joint_xyz']
                    train_set[(subj, action, subact)] = train_set[(subj, action, subact)].reshape(
                        train_set[(subj, action, subact)].shape[0], -1)

            if len(complete_train) == 0:
                complete_train = copy.deepcopy(train_set[(subj, action, subact)])
            else:
                complete_train = np.append(complete_train, train_set[(subj, action, subact)], axis=0)

    test_set = {}
    complete_test = []
    for subj in [5]:
        for action in actions:
            for subact in [1, 2]:
                if config.datatype == 'lie':
                    filename = '{0}/S{1}_{2}_{3}_lie.mat'.format('./data/h3.6m/Test/test_lie', subj, action, subact)
                    test_set[(subj, action, subact)] = sio.loadmat(filename)['lie_parameters']

                if config.datatype == 'xyz':
                    filename = '{0}/S{1}_{2}_{3}_xyz.mat'.format('./data/h3.6m/Test/test_xyz', subj, action, subact)
                    test_set[(subj, action, subact)] = sio.loadmat(filename)['joint_xyz']
                    test_set[(subj, action, subact)] = test_set[(subj, action, subact)].reshape(
                        test_set[(subj, action, subact)].shape[0], -1)

            if len(complete_test) == 0:
                complete_test = copy.deepcopy(test_set[(subj, action, subact)])
            else:
                complete_test = np.append(complete_test, test_set[(subj, action, subact)], axis=0)

    if config.datatype == 'lie':
        # Compute normalization stats
        data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)
        # The global translation and rotation are not considered since we perform procrustes alignment
        # dim_to_ignore = [0,1,2,3,4,5] + dim_to_ignore
        # dim_to_use = dim_to_use[6:]
        config.data_mean = data_mean
        config.data_std = data_std
        config.dim_to_ignore = dim_to_ignore
        config.dim_to_use = dim_to_use

        # Normalize: subtract mean, divide by std
        train_set = data_utils.normalize_data(train_set, data_mean, data_std, dim_to_use)
        test_set = data_utils.normalize_data(test_set, data_mean, data_std, dim_to_use)

        expmapInd = np.split(np.arange(4, 100) - 1, 32)

        weights = np.zeros([len(config.dim_to_use)])
        for j in range(len(config.dim_to_use)):
            for i in range(len(expmapInd)):
                if config.dim_to_use[j] in expmapInd[i]:
                    weights[j] = i + 1
                    break
        weights = list(map(int, weights))

        chain = [[0], [132.95, 442.89, 454.21, 162.77, 75], [132.95, 442.89, 454.21, 162.77, 75],
                 [132.95, 253.38, 257.08, 121.13, 115], [0, 151.03, 278.88, 251.73, 100, 0, 0, 0],
                 [0, 151.03, 278.88, 251.73, 100, 0, 0, 0]]
        for x in chain:
            s = sum(x)
            if s == 0:
                continue
            for i in range(len(x)):
                x[i] = (i+1)*sum(x[i:])/s

        chain = [item for sublist in chain for item in sublist]

        config.weights = []
        for i in range(len(weights)):
            config.weights.append(chain[weights[i]])

    config.input_size = train_set[list(train_set.keys())[0]].shape[1]

    x_test = {}
    y_test = {}
    dec_in_test = {}
    for action in actions:
        encoder_inputs, decoder_inputs, decoder_outputs = get_batch_srnn(config, test_set, action, seq_length_out)
        x_test[action] = encoder_inputs
        y_test[action] = decoder_outputs
        dec_in_test[action] = np.zeros(decoder_inputs.shape)
        dec_in_test[action][:, 0, :] = decoder_inputs[:, 0, :]

    print("Done reading data.")

    return [train_set, test_set, x_test, y_test, dec_in_test, config]


def read_animals(config, training, train_subjects, test_subjects, train_path, x_test_path, y_test_path):

    seq_length_in = config.input_window_size

    if training:
        seq_length_out = config.output_window_size
    else:
        seq_length_out = config.test_output_window

    # Read a base file to obtain bone lengths
    if config.datatype == 'lie':
        filename = y_test_path + 'test_0' + '_lie.mat'
        rawdata = sio.loadmat(filename)
        matdict_key = list(rawdata.keys())[3]
        rawdata = rawdata[matdict_key]
        njoints = rawdata.shape[1]
        bone = np.zeros([njoints, 3])

        # Bone lengths
        for i in range(njoints):
            bone[i, 0] = round(rawdata[0, i, 3], 2)

    elif config.datatype == 'xyz':
        filename = y_test_path + 'test_0' + '.mat'
        rawdata = sio.loadmat(filename)
        matdict_key = list(rawdata.keys())[3]
        rawdata = rawdata[matdict_key]
        njoints = rawdata.shape[1]
        bone = np.zeros([njoints, 3])

        # Bone lengths
        for i in range(1,njoints):
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
            rawdata = sio.loadmat(filename)
            matdict_key = list(rawdata.keys())[3]
            rawdata = rawdata[matdict_key]
            data = rawdata[:, :-1, :3].reshape(rawdata.shape[0], -1)
            train_set[id] = data

        elif config.datatype == 'xyz':
            filename = train_path + id + '_xyz.mat'
            rawdata = sio.loadmat(filename)
            matdict_key = list(rawdata.keys())[3]
            rawdata = rawdata[matdict_key]
            data = rawdata.reshape(rawdata.shape[0], -1)
            train_set[id] = data

    test_set = {}

    for id in test_subjects:

        if config.datatype == 'lie':
            filename = train_path + id + '_lie.mat'
            rawdata = sio.loadmat(filename)
            matdict_key = list(rawdata.keys())[3]
            rawdata = rawdata[matdict_key]
            data = rawdata[:, :-1, :3].reshape(rawdata.shape[0], -1)
            test_set[id] = data

        elif config.datatype == 'xyz':
            filename = train_path + id + '_xyz.mat'
            rawdata = sio.loadmat(filename)
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

            x_rawdata = sio.loadmat(x_filename)
            matdict_key = list(x_rawdata.keys())[3]
            x_rawdata = x_rawdata[matdict_key]

            y_rawdata = sio.loadmat(y_filename)
            matdict_key = list(y_rawdata.keys())[3]
            y_rawdata = y_rawdata[matdict_key]

            x_data = x_rawdata[:, :-1, :3].reshape(x_rawdata.shape[0], -1)
            x_test.append(x_data)

            y_data = y_rawdata[:, :-1, :3].reshape(y_rawdata.shape[0], -1)
            y_test.append(y_data)

        elif config.datatype == 'xyz':
            x_filename = x_test_path + 'test_' + str(i) + '.mat'
            y_filename = y_test_path + 'test_' + str(i) + '.mat'

            x_rawdata = sio.loadmat(x_filename)
            matdict_key = list(x_rawdata.keys())[3]
            x_rawdata = x_rawdata[matdict_key]

            y_rawdata = sio.loadmat(y_filename)
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

    train_set, test_set, x_test_dict, y_test_dict, dec_in_test_dict, config = read_animals(config, training, train_subjects, test_subjects, train_path, x_test_path, y_test_path)

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

    train_set, test_set, x_test_dict, y_test_dict, dec_in_test_dict, config = read_animals(config, training, train_subjects, test_subjects, train_path, x_test_path, y_test_path)

    return [train_set, test_set, x_test_dict, y_test_dict, dec_in_test_dict, config]


def get_batch_srnn(config, data, action, target_seq_len):
    # Obtain SRNN test sequences using the specified random seeds

    frames = {}
    frames[action] = find_indices_srnn( data, action )

    batch_size = 8
    subject = 5
    source_seq_len = config.input_window_size

    seeds = [(action, (i%2)+1, frames[action][i]) for i in range(batch_size)]

    encoder_inputs = np.zeros((batch_size, source_seq_len-1, config.input_size), dtype=float )
    decoder_inputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float )
    decoder_outputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float )

    for i in range(batch_size):
        _, subsequence, idx = seeds[i]
        idx = idx + 50

        data_sel = data[(subject, action, subsequence)]

        data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len), :]

        encoder_inputs[i, :, :] = data_sel[0:source_seq_len-1, :] #x_test
        decoder_inputs[i, :, :] = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :] #decoder_in_test
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

    T1 = data[(subject, action, subaction1)].shape[0]
    T2 = data[(subject, action, subaction2)].shape[0]
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
