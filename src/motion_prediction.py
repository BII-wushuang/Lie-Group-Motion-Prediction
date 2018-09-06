import numpy as np
import tensorflow as tf
import os
import training_config
from read_data import read_data
import models
import loss_functions
from plot_animation import plot_animation
from general_utils import Progbar, create_directory
import re
import scipy.io as sio
import general_utils as data_utils
import timeit



tf.app.flags.DEFINE_string("dataset", "Human", "Articulate object dataset: 'Human' or 'Fish' or 'Mouse'.")
tf.app.flags.DEFINE_string("datatype", "lie", "Datatype can be 'lie' or 'xyz'.")
tf.app.flags.DEFINE_string("action", "all", "Action is 'default' for 'Fish' and 'Mouse' and one or all of the following for 'Human'.")
'''
h3.6m_action_list = ["directions", "discussion", "eating", "greeting", "phoning",
          "posing", "purchases", "sitting", "sittingdown", "smoking",
          "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
'all' includes all of the above
mouse/fish_action = 'default'
'''
tf.app.flags.DEFINE_boolean("training", True, "Set to True for training.")
tf.app.flags.DEFINE_boolean("visualize", True, "Set to True for visualization.")
tf.app.flags.DEFINE_boolean("longterm", False, "Set to True for super long-term prediction.")  #if longterm is true, action only can be: 'walking', 'eating' or 'smoking'

FLAGS = tf.app.flags.FLAGS

def train():
    print("Training")

    # tf Graph input
    x = tf.placeholder(dtype=tf.float32, shape=[None, config.input_window_size - 1, config.input_size], name="input_sequence")
    y = tf.placeholder(dtype=tf.float32, shape=[None, config.output_window_size, config.input_size], name="raw_labels")
    dec_in = tf.placeholder(dtype=tf.float32, shape=[None, config.output_window_size, config.input_size], name="decoder_input")

    labels = tf.transpose(y, [1, 0, 2])
    labels = tf.reshape(labels, [-1, config.input_size])
    labels = tf.split(labels, config.output_window_size, axis=0, name='labels')

    tf.set_random_seed(112858)

    # Define model
    prediction = models.seq2seq(x, dec_in, config, True)

    sess = tf.Session()

    # Define cost function
    loss = eval('loss_functions.' + config.loss + '_loss(prediction, labels, config)')

    # Add a summary for the loss
    train_loss = tf.summary.scalar('train loss', loss)
    valid_loss = tf.summary.scalar('valid loss', loss)

    # Defining training parameters
    optimizer = tf.train.AdamOptimizer(config.learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Gradient Clipping
    grads = tf.gradients(loss, tf.trainable_variables())
    grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)
    optimizer.apply_gradients(zip(grads, tf.trainable_variables()))
    train_op = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=5)
    train_writer = tf.summary.FileWriter("./log", sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Obtain total training parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Total training parameters: ' + str(total_parameters))

    if not(os.path.exists(checkpoint_dir)):
        os.makedirs(checkpoint_dir)

    saved_epoch = 0
    if config.restore & os.path.exists(checkpoint_dir+'checkpoint'):
        with open(checkpoint_dir + 'checkpoint') as f:
            content = f.readlines()
        saved_epoch = int(re.search(r'\d+', content[0]).group())
        model_name = checkpoint_dir + "Epoch_" + str(saved_epoch)
        saver.restore(sess, model_name)
        print("Restored session from Epoch ", str(saved_epoch))

    print("________________________________________________________________")

    train_size = config.training_size
    valid_size = config.validation_size
    best_val_loss = float('inf')
    best_val_epoch = saved_epoch

    for j in range(saved_epoch, config.max_epoch):

        print("Epoch ", j+1)
        prog = Progbar(target=train_size)
        prog_valid = Progbar(target=valid_size)

        for i in range(train_size):
            batch_x, batch_dec_in, batch_y = data_utils.get_batch(config, train_set)
            current_cost, train_summary, _ = sess.run([loss, train_loss, train_op], feed_dict={x: batch_x, y: batch_y, dec_in: batch_dec_in})

            train_writer.add_summary(train_summary, j*train_size+i)
            prog.update(i+1, [("Training Loss", current_cost)])

        v_loss_mean = 0.0
        for i in range(valid_size):
            batch_x, batch_dec_in, batch_y = data_utils.get_batch(config, test_set)
            v_loss, valid_summary = sess.run([loss, valid_loss], feed_dict={x: batch_x, y: batch_y, dec_in: batch_dec_in})
            v_loss_mean = v_loss_mean*i/(i+1) + v_loss/(i+1)
            prog_valid.update(i + 1, [("Validation Loss", v_loss)])
            train_writer.add_summary(valid_summary, j*valid_size+i)

        if v_loss_mean < best_val_loss:
            model_name = checkpoint_dir + "Epoch_" + str(j+1)
            best_val_loss = v_loss_mean
            best_val_epoch = j+1
            saver.save(sess, model_name)

        print("Current Best Epoch: ", best_val_epoch, ", Best Validation Loss: ", best_val_loss, "\n")

        if j+1 - best_val_epoch > config.early_stop:
            break


def predict():
    print("Predicting")

    tf.reset_default_graph()

    # tf Graph input
    x = tf.placeholder(dtype=tf.float32, shape=[None, config.input_window_size - 1, config.input_size], name="input_sequence")
    dec_in = tf.placeholder(dtype=tf.float32, shape=[None, config.test_output_window, config.input_size], name="decoder_input")

    tf.set_random_seed(112858)

    # Define model
    prediction = models.seq2seq(x, dec_in, config, False)

    sess = tf.Session()

    # Restore latest model
    with open(checkpoint_dir + 'checkpoint') as f:
        content = f.readlines()
    saved_epoch = int(re.search(r'\d+', content[0]).group())
    model_name = checkpoint_dir + "Epoch_" + str(saved_epoch)
    saver = tf.train.Saver()
    saver.restore(sess, model_name)
    print("Restored session from Epoch ", str(saved_epoch))

    start = timeit.default_timer()

    y_predict = {}
    for act in actions:
        pred = sess.run(prediction, feed_dict={x: x_test[act], dec_in: dec_in_test[act]})
        pred = np.array(pred)
        pred = np.transpose(pred, [1, 0, 2])
        y_predict[act] = pred

    stop = timeit.default_timer()
    print("Test Time: ", stop - start)

    return y_predict


def main(_):

    global config, actions, checkpoint_dir, output_dir, train_set, test_set, x_test, y_test, dec_in_test

    config = training_config.train_Config(FLAGS.dataset, FLAGS.datatype, FLAGS.action)

    if FLAGS.longterm == True:
        config.output_window_size = 100
        if FLAGS.action not in ['walking', 'eating', 'smoking']:
            raise Exception("Invalid action! For long-term prediction, action can only be 'walking', 'smoking' or 'eating'.")

    # Define checkpoint & output directory
    checkpoint_dir, output_dir = create_directory(config)

    # Train model
    if FLAGS.training:
        train_set, test_set, x_test, y_test, dec_in_test, config = read_data(config, True)
        actions = list(x_test.keys())
        train()

    # Predict on test set with trained model
    try: x_test
    except NameError: x_test = None
    if config.test_output_window > config.output_window_size or x_test is None:
        train_set, test_set, x_test, y_test, dec_in_test, config = read_data(config, False)
        actions = list(x_test.keys())

    if FLAGS.longterm is True:
        x_test = {}
        y_test = {}
        dec_in_test = {}
        test_set = test_set[list(test_set.keys())[0]]
        x_test[FLAGS.action] = np.reshape(test_set[:config.input_window_size-1,:], [1, -1, config.input_size])
        y_test[FLAGS.action] = np.reshape(test_set[config.input_window_size:, :], [1, -1, config.input_size])
        dec_in_test[FLAGS.action] = np.reshape(test_set[config.input_window_size-1:-1, :], [1, -1, config.input_size])
        config.test_output_window = y_test[FLAGS.action].shape[1]
        config.batch_size = 1
        actions = [FLAGS.action]
        test_actions = [FLAGS.action]
    else:
        test_actions = actions

    y_predict = predict()

    if not (os.path.exists(output_dir)):
        os.makedirs(output_dir)
    print("Outputs saved to: " + output_dir)

    for action in test_actions:

        if config.datatype == 'lie':
            mean_error, _ = data_utils.mean_euler_error(config, action, y_predict[action], y_test[action])
            sio.savemat(output_dir + 'error_' + action + '.mat', dict([('error', mean_error)]))

            for i in range(y_predict[action].shape[0]):
                if config.dataset == 'Human':
                    y_p = data_utils.unNormalizeData(y_predict[action][i], config.data_mean, config.data_std, config.dim_to_ignore)
                    y_t = data_utils.unNormalizeData(y_test[action][i], config.data_mean, config.data_std, config.dim_to_ignore)
                    expmap_all = data_utils.revert_coordinate_space(np.vstack((y_t, y_p)), np.eye(3), np.zeros(3))
                    y_p = expmap_all[config.test_output_window:]
                    y_t = expmap_all[:config.test_output_window]
                else:
                    y_p = y_predict[action][i]
                    y_t = y_test[action][i]

                sio.savemat(output_dir + 'prediction_lie_' + action + '_' + str(i) + '.mat', dict([('prediction', y_p)]))
                sio.savemat(output_dir + 'gt_lie_' + action + '_' + str(i) + '.mat', dict([('gt', y_t)]))

                # Forward Kinematics to obtain 3D xyz locations
                y_p_xyz = data_utils.fk(y_p, config)
                y_t_xyz = data_utils.fk(y_t, config)

                sio.savemat(output_dir + 'prediction_xyz_' + action + '_' + str(i) + '.mat', dict([('prediction', y_p_xyz)]))
                sio.savemat(output_dir + 'gt_xyz_' + action + '_' + str(i) + '.mat', dict([('gt', y_t_xyz)]))

                filename = action + '_' + str(i)
                if FLAGS.visualize:
                    # Visualize prediction
                    predict_plot = plot_animation(y_p_xyz, y_t_xyz, config, filename)
                    predict_plot.plot()

        else:
            for i in range(y_predict[action].shape[0]):
                y_p = y_predict[action][i]
                y_t = y_test[action][i]

                y_p_xyz = np.reshape(y_p, [y_p.shape[0], -1, 3])
                y_t_xyz = np.reshape(y_t, [y_t.shape[0], -1, 3])

                sio.savemat(output_dir + 'prediction_xyz_' + action + '_' + str(i) + '.mat', dict([('prediction', y_p)]))
                sio.savemat(output_dir + 'gt_xyz_' + action + '_' + str(i) + '.mat', dict([('gt', y_t)]))

                # Inverse Kinematics to obtain lie parameters
                y_p_lie = data_utils.inverse_kinematics(y_p, config)
                y_t_lie = data_utils.inverse_kinematics(y_t, config)

                sio.savemat(output_dir + 'prediction_lie_' + action + str(i) + '.mat', dict([('prediction', y_p_lie)]))
                sio.savemat(output_dir + 'gt_lie_' + action + str(i) + '.mat', dict([('gt', y_t_lie)]))

                filename = action + '_' + str(i)
                if FLAGS.visualize:
                    # Visualize prediction
                    predict_plot = plot_animation(y_p_xyz, y_t_xyz, config, filename)
                    predict_plot.plot()


if __name__ == '__main__':
    # Load dataset and training parameters
    tf.app.run()
