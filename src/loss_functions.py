import tensorflow as tf
import numpy as np


def l2_loss(prediction, y, config):
    # L2 Loss
    loss = tf.square(tf.subtract(y, prediction))
    loss = tf.reduce_mean(loss)

    return loss


def linearizedlie_loss(prediction, y, config):
    # Linearized geodesic loss
    chainlength = config.bone.shape[0] - 1
    weights = np.zeros([chainlength * 3])
    for j in range(chainlength):
        for i in range(j, chainlength):
            weights[j*3:j*3+3] = weights[j*3] + (chainlength - i) * config.bone[i + 1][0]
    weights = weights/weights.max()
    loss = tf.square(tf.subtract(y,prediction))
    loss = tf.reduce_mean(loss, axis=[0,1])
    loss = tf.reduce_mean(tf.multiply(loss,weights))

    return loss


def lie_loss(prediction, y, config):
    # Compute the joint discrepancy following forward kinematics of lie parameters
    prediction = tf.concat(prediction, axis=0)
    y = tf.concat(y, axis=0)
    joint_pred = forward_kinematics(prediction, config)
    joint_label = forward_kinematics(y, config)
    loss = tf.reduce_mean(tf.square(tf.subtract(joint_pred, joint_label)))

    return loss


def angle_loss(prediction, y, config):
    # Compute the geodesic of the rotational loss on SE(3)
    n_joints = int(config.input_size / 3)
    y_batch = tf.split(y, num_or_size_splits=config.batch_size, axis=0)
    prediction_batch = tf.split(prediction, num_or_size_splits=config.batch_size, axis=0)
    R_loss = tf.zeros([])
    for h in range(config.batch_size):
        y_split = tf.split(y_batch[h], num_or_size_splits=n_joints, axis=1)
        prediction_split = tf.split(prediction_batch[h], num_or_size_splits=n_joints, axis=1)
        for i in range(1, n_joints):
            R_y = rotmat(tf.squeeze(y_split[i], axis=0))
            R_prediction = rotmat(tf.squeeze(prediction_split[i], axis=0))
            R = tf.matmul(R_prediction, tf.matrix_transpose(R_y))
            R_loss = R_loss + tf.square(tf.acos(((tf.trace(R)-1)/2)))
    loss = 2 * tf.sqrt(R_loss / config.batch_size)

    return loss


def rotmat(v):
    v_norm = tf.norm(v)
    v = v / v_norm
    v_split = tf.split(v, num_or_size_splits=3)
    v_cross = tf.stack([[tf.zeros(1,1), -v_split[2], v_split[1]], [v_split[2], tf.zeros(1,1), -v_split[0]], [-v_split[1], v_split[0], tf.zeros(1,1)]])
    v_cross = tf.squeeze(v_cross, axis=2)
    R = tf.cond(v_norm < 1e-5, lambda: tf.eye(3), lambda: tf.eye(3) + tf.sin(v_norm) * v_cross + (1-tf.cos(v_norm)) * tf.matmul(v_cross, v_cross))
    return R


def forward_kinematics(lie_parameters, config):
    nframes = lie_parameters.get_shape().as_list()[0]
    lie_parameters = tf.reshape(lie_parameters, [nframes, -1, 3])

    R = []
    idx = config.idx
    chain_idx = config.chain_idx
    bone_params = config.bone_params
    for h in range(nframes):
        omega = []
        A = []
        chain = []
        for i in range(len(idx) - 1):
            chain.append(tf.concat([lie_parameters[h, idx[i]:idx[i + 1]], tf.zeros([1, 3])], axis=0))

        omega.append(tf.concat(chain, axis=0))

        for i in range(omega[0].shape[0]):
            A.append([rotmat(omega[0][i])])
        R.append(tf.concat(A, axis=0))

    R = tf.stack(R)
    joints = []
    for h in range(nframes):
        jointlist = []
        for i in range(len(chain_idx)):
            for j in range(chain_idx[i].shape[0]):
                if j == 0:
                    if i < 3:
                        jointlist.append(tf.zeros([3, 1]))
                    else:
                        jointlist.append(joint_xyz[14])
                else:
                    k = j - 1
                    A = R[h, chain_idx[i][k]]
                    while k > 0:
                        k = k - 1
                        A = tf.matmul(R[h, chain_idx[i][k]], A)
                    jointlist.append(
                        tf.matmul(A, tf.reshape(bone_params[chain_idx[i][j]], [3, 1])) + joint_xyz[chain_idx[i][j - 1]])
                joint_xyz = tf.stack(jointlist)
        joints.append(tf.squeeze(joint_xyz))
    joints = tf.stack(joints)
    return joints


def findrot(u, v):
    w = tf.cross(u, v)
    w_norm = tf.norm(w)
    axisangle = tf.cond(w_norm < 1e-5, lambda: tf.zeros([3]), lambda: w / w_norm * tf.acos(tf.tensordot(u, v, axes=1)))
    return axisangle


def axis_angle(R):
    theta = tf.acos((tf.trace(R) - 1) / 2.0)
    omega = tf.cond(theta < 1e-5, lambda: tf.zeros([3]), lambda: theta / (2 * tf.sin(theta)) * tf.stack(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], axis=0))
    return omega


def inverse_kinematics(joint_xyz, config):
    nframes = joint_xyz.get_shape().as_list()[0]
    joint_xyz = tf.reshape(joint_xyz, [nframes, -1, 3])

    chain_idx = config.chain_idx
    lie_parameters = []
    lie_parameters_list = []
    for i in range(nframes):
        for k in range(len(chain_idx) - 1, -1, -1):
            for j in range(len(chain_idx[k]) - 2, -1, -1):
                v = tf.squeeze(joint_xyz[i, chain_idx[k][j + 1], :] - joint_xyz[i, chain_idx[k][j], :])
                vhat = v / tf.norm(v)
                if j == 0:
                    uhat = tf.constant([1., 0, 0])
                else:
                    u = tf.squeeze(joint_xyz[i, chain_idx[k][j], :] - joint_xyz[i, chain_idx[k][j - 1], :])
                    uhat = u / tf.norm(u)

                A = tf.transpose(rotmat(findrot(tf.constant([1., 0, 0]), tf.squeeze(uhat))))
                B = rotmat(findrot(tf.constant([1., 0, 0]), tf.squeeze(vhat)))
                lie_parameters = [tf.squeeze(axis_angle(tf.matmul(A, B)))] + lie_parameters
        lie_parameters = tf.stack(lie_parameters)
        lie_parameters_list.append(lie_parameters)
    lie_parameters_list = tf.stack(lie_parameters_list)
    return lie_parameters_list


def forward_kinematics_euler(lie_parameters, config):
    nframes = lie_parameters.get_shape().as_list()[0]
    lie_parameters = tf.reshape(lie_parameters, [nframes, -1, 3])

    R = []
    idx = config.idx
    chain_idx = config.chain_idx
    bone_params = config.bone_params
    for h in range(nframes):
        omega = []
        A = []
        chain = []
        for i in range(len(idx) - 1):
            chain.append(tf.concat([lie_parameters[h, idx[i]:idx[i + 1]], tf.zeros([1, 3])], axis=0))

        omega.append(tf.concat(chain, axis=0))

        for i in range(omega[0].shape[0]):
            A.append([euler(omega[0][i])])
        R.append(tf.concat(A, axis=0))

    R = tf.stack(R)
    joints = []
    for h in range(nframes):
        jointlist = []
        for i in range(len(chain_idx)):
            for j in range(chain_idx[i].shape[0]):
                if j == 0:
                    if i < 3:
                        jointlist.append(tf.zeros([3, 1]))
                    else:
                        jointlist.append(joint_xyz[14])
                else:
                    k = j - 1
                    A = R[h, chain_idx[i][k]]
                    while k > 0:
                        k = k - 1
                        A = tf.matmul(R[h, chain_idx[i][k]], A)
                    jointlist.append(
                        tf.matmul(A, tf.reshape(bone_params[chain_idx[i][j]], [3, 1])) + joint_xyz[chain_idx[i][j - 1]])
                joint_xyz = tf.stack(jointlist)
        joints.append(tf.squeeze(joint_xyz))
    joints = tf.stack(joints)
    return joints


def euler(angle):
    a = angle[0]
    b = angle[1]
    c = angle[2]
    a1 = tf.constant([1., 0, 0])
    a2 = tf.stack([tf.constant(0.), tf.cos(a), -tf.sin(a)])
    a3 = tf.stack([tf.constant(0.), tf.sin(a), tf.cos(a)])

    A = tf.stack([a1, a2, a3])

    b1 = tf.stack([tf.cos(b), tf.constant(0.), tf.sin(b)])
    b2 = tf.constant([0., 1, 0])
    b3 = tf.stack([-tf.sin(b), tf.constant(0.), tf.cos(b)])

    B = tf.stack([b1, b2, b3])

    c1 = tf.stack([tf.cos(c), -tf.sin(c), tf.constant(0.)])
    c2 = tf.stack([tf.sin(c), tf.cos(c), tf.constant(0.)])
    c3 = tf.constant([0., 0, 1])

    C = tf.stack([c1, c2, c3])

    R = tf.matmul(tf.matmul(A, B), C)
    return R
