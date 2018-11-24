import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


def lf(prev, i):  # function for seq2seq recurrent loss
    return prev


def seq2seq(input, dec_in, config, training):
    seq_length_in = config.input_window_size

    if training:
        seq_length_out = config.output_window_size
    else:
        seq_length_out = config.test_output_window

    # Inputs
    enc_in = tf.transpose(input, [1, 0, 2])
    dec_in = tf.transpose(dec_in, [1, 0, 2])

    enc_in = tf.reshape(enc_in, [-1, config.input_size])
    dec_in = tf.reshape(dec_in, [-1, config.input_size])

    enc_in = tf.split(enc_in, seq_length_in-1, axis=0)
    dec_in = tf.split(dec_in, seq_length_out, axis=0)

    if config.model == 'ERD':
        # Encoder
        fc = [tf.layers.dense(enc_in[i], 500,activation= tf.nn.relu,reuse=tf.AUTO_REUSE, name="fc") for i in range(config.input_window_size-1)]
        config.hidden_size = 1000
        hidden_size = [config.hidden_size, config.hidden_size]
        number_of_layers = len(hidden_size)

        def lstm_cell(size):
            cell = tf.contrib.rnn.LSTMCell(size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            return cell

        enc_cells = [lstm_cell(hidden_size[i]) for i in range(number_of_layers)]
        enc_cell = tf.contrib.rnn.MultiRNNCell(enc_cells)
        output, final_state = tf.contrib.rnn.static_rnn(enc_cell, fc, dtype=tf.float32)
        enc_state = [(final_state[i][0], final_state[i][1]) for i in range(number_of_layers)]

        # Decoder
        dec_cell = [tf.nn.rnn_cell.LSTMCell(hidden_size[i]) for i in range(number_of_layers)]
        dec_cell = tf.contrib.rnn.MultiRNNCell(dec_cell)
        dec_cell = ERDWrapper(dec_cell, config)

    elif config.model == 'LSTM3lr':
        # Encoder
        config.hidden_size = 1000
        hidden_size = [config.hidden_size, config.hidden_size, config.hidden_size]
        number_of_layers = len(hidden_size)

        def lstm_cell(size):
            cell = tf.contrib.rnn.LSTMCell(size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            return cell

        enc_cells = [lstm_cell(hidden_size[i]) for i in range(number_of_layers)]
        enc_cell = tf.contrib.rnn.MultiRNNCell(enc_cells)
        output, final_state = tf.contrib.rnn.static_rnn(enc_cell, enc_in, dtype=tf.float32)
        enc_state = [(final_state[i][0], final_state[i][1]) for i in range(number_of_layers)]

        # Decoder
        dec_cell = [tf.nn.rnn_cell.LSTMCell(hidden_size[i]) for i in range(number_of_layers)]
        dec_cell = tf.contrib.rnn.MultiRNNCell(dec_cell)
        dec_cell = StackedLSTMWrapper(dec_cell, config)

    elif config.model == 'GRU':
        # Encoder
        config.hidden_size = 1024
        enc_cell = tf.contrib.rnn.GRUCell(config.hidden_size)
        _, enc_state = tf.contrib.rnn.static_rnn(enc_cell, enc_in, dtype=tf.float32)  # Encoder

        # Decoder
        dec_cell = tf.contrib.rnn.GRUCell(config.hidden_size)
        dec_cell = HMRWrapper(dec_cell, config)

    elif config.model == 'HMR':
        # Linear embedding weights HMR
        weights_in = tf.get_variable("encoder_w", [config.input_size, config.hidden_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
        bias_in = tf.get_variable("encoder_b", [config.hidden_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
        
        h = tf.matmul(tf.reshape(input, [-1, config.input_size]), weights_in) + bias_in
        h = tf.nn.dropout(h, config.keep_prob)
        h = tf.reshape(h, [-1, config.input_window_size-1, config.hidden_size])

        c_h = tf.identity(h)
        c_h = tf.nn.dropout(c_h, config.keep_prob)

        # Encoder
        hidden_states, cell_states, global_state = hmr_cell(h, c_h, config)
        enc_state = [(tf.reduce_mean(hidden_states[-1], axis=1), tf.reduce_mean(cell_states[-1], axis=1)), (tf.reduce_mean(tf.concat([hidden_states[-1], tf.expand_dims(global_state[-1], axis=1)], axis=1), axis=1), tf.reduce_mean(cell_states[-1], axis=1))]

        # Decoder
        dec_cell = [tf.nn.rnn_cell.LSTMCell(config.hidden_size, name='decoder_cell_' + str(i)) for i in range(2)]
        dec_cell = tf.contrib.rnn.MultiRNNCell(dec_cell)
        dec_cell = HMRWrapper(dec_cell, config)

    with tf.variable_scope("basic_rnn_seq2seq"):
        outputs, states = tf.contrib.legacy_seq2seq.rnn_decoder(dec_in, enc_state, dec_cell, loop_function=lf)
    return outputs


class ERDWrapper(RNNCell):
    # ERD decoder wrapper
    def __init__(self, cell, config):
        self.cell = cell
        self.config = config

    def __call__(self, inputs, state, scope=None):

        fc_in_1 = tf.layers.dense(inputs, 500, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="fc_in_1")
        fc_in_2 = tf.layers.dense(fc_in_1, 500, activation=None, reuse=tf.AUTO_REUSE, name="fc_in_2")

        output, new_state = self.cell(fc_in_2, state, scope)

        fc1 = tf.layers.dense(output, 500, activation= tf.nn.relu,reuse=tf.AUTO_REUSE, name="fc1")
        fc2 = tf.layers.dense(fc1, 100, activation= tf.nn.relu,reuse=tf.AUTO_REUSE, name="fc2")

        out = tf.layers.dense(fc2, self.config.input_size, reuse=tf.AUTO_REUSE, name="output")

        return out, new_state


class StackedLSTMWrapper(RNNCell):
    # StackedLSTM decoder wrapper
    def __init__(self, cell, config):
        self.cell = cell
        self.config = config

        insize = config.hidden_size

        output_size = config.input_size
        with tf.variable_scope("decoder_weights", reuse=tf.AUTO_REUSE):
            self.w_out = tf.get_variable("decoder_w", [insize, output_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
            self.b_out = tf.get_variable("decoder_b", [output_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.config.input_size

    def __call__(self, inputs, state, scope=None):
        output, new_state = self.cell(inputs, state, scope)
        output = tf.matmul(output, self.w_out) + self.b_out

        return output, new_state


class HMRWrapper(RNNCell):
    # HMR / GRU decoder wrapper
    def __init__(self, cell, config):

        self.cell = cell
        self.config = config

        insize = config.hidden_size

        output_size = config.input_size
        with tf.variable_scope("decoder_weights", reuse=tf.AUTO_REUSE):
            self.w_out = tf.get_variable("decoder_w", [insize, output_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
            self.b_out = tf.get_variable("decoder_b", [output_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.config.input_size

    def __call__(self, inputs, state, scope=None):

        output, new_state = self.cell(inputs, state, scope)
        output = tf.matmul(output, self.w_out) + self.b_out
        output = tf.add(output, inputs)

        if self.config.datatype == 'xyz':
            output = xyz_resize(output, self.config)

        return output, new_state


def get_hidden_states_before(hidden_states, step, padding):
    displaced_hidden_states=hidden_states[:,:-step,:]
    return tf.concat([padding, displaced_hidden_states], axis=1)


def get_hidden_states_after(hidden_states, step, padding):
    displaced_hidden_states=hidden_states[:,step:,:]
    return tf.concat([displaced_hidden_states, padding], axis=1)


def sum_together(l):
    combined_state=None
    for tensor in l:
        if combined_state==None:
            combined_state=tensor
        else:
            combined_state=combined_state+tensor
    return combined_state


def hmr_cell(h, c_h, config):
    hidden_size = config.hidden_size
    rec_steps = config.recurrent_steps

    with tf.name_scope("HMR"):
        with tf.variable_scope("HMR", reuse=tf.AUTO_REUSE):
            '''h update gates'''
            # forward forget gate
            with tf.name_scope("f_gate"):
                Uf = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Uf")
                Wlrf = tf.get_variable(initializer=tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wlrf")
                Wf = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wf")
                Zf = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Zf")
                bf = tf.get_variable(initializer=tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf")

            # left forget gate
            with tf.name_scope("l_gate"):
                Ul = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Ul")
                Wlrl = tf.get_variable(initializer=tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wlrl")
                Wl = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wl")
                Zl = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Zl")
                bl = tf.get_variable(initializer=tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bl")

            # right forget gate
            with tf.name_scope("r_gate"):
                Ur = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Ur")
                Wlrr = tf.get_variable(initializer=tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wlrr")
                Wr = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wr")
                Zr = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Zr")
                br = tf.get_variable(initializer=tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="br")

            # forget gate for g
            with tf.name_scope("q_gate"):
                Uq = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Uq")
                Wlrq = tf.get_variable(initializer=tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wlrq")
                Wq = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wq")
                Zq = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Zq")
                bq = tf.get_variable(initializer=tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bq")

            # input gate
            with tf.name_scope("i_gate"):
                Ui = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Ui")
                Wlri = tf.get_variable(initializer=tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wlri")
                Wi = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wi")
                Zi = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Zi")
                bi = tf.get_variable(initializer=tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bi")

            # output gate
            with tf.name_scope("o_gate"):
                Uo = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Uo")
                Wlro = tf.get_variable(initializer=tf.random_normal([2 * hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wlro")
                Wo = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wo")
                Zo = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Zo")
                bo = tf.get_variable(initializer=tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")

            '''g update gates'''
            # forget gates for h
            with tf.name_scope("g_f_gate"):
                g_Wf = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="g_Wf")
                g_Zf = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="g_Zf")
                g_bf = tf.get_variable(initializer=tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="g_bf")

            # forget gate for g
            with tf.name_scope("g_g_gate"):
                g_Wg = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="g_Wg")
                g_Zg = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="g_Zg")
                g_bg = tf.get_variable(initializer=tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="g_bg")

            # output gate
            with tf.name_scope("g_o_gate"):
                g_Wo = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="g_Wo")
                g_Zo = tf.get_variable(initializer=tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="g_Zo")
                g_bo = tf.get_variable(initializer=tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="g_bo")


            # record shape of the batch
            shape = h.get_shape().as_list()
            # shape[0] = config.batch_size

            # inital g
            g = tf.reduce_mean(h, axis=1)
            c_g = tf.reduce_mean(c_h, axis=1)

            final_hidden_states = []
            final_cell_states = []
            final_global_states = []

            padding = tf.zeros_like(h[:,0:config.context_window,:])

            for i in range(rec_steps):

                '''Update g'''
                g_tilde = tf.reduce_mean(h, axis=1)

                # reshape for matmul
                reshaped_h = tf.reshape(h, [-1, hidden_size])
                reshaped_g = tf.reshape(tf.tile(tf.expand_dims(g, axis=1), [1, shape[1], 1]), [-1, hidden_size])

                # forget gates for g
                g_f_n = tf.nn.sigmoid(tf.matmul(reshaped_g, g_Zf) + tf.matmul(reshaped_h, g_Wf) + g_bf)
                g_g_n = tf.nn.sigmoid(tf.matmul(g, g_Zg) + tf.matmul(g_tilde, g_Wg) + g_bg)
                g_o_n = tf.nn.sigmoid(tf.matmul(g, g_Zo) + tf.matmul(g_tilde, g_Wo) + g_bo)

                # reshape the gates
                reshaped_g_f_n = tf.reshape(g_f_n, [-1, shape[1], hidden_size])
                reshaped_g_g_n = tf.reshape(g_g_n, [-1, 1, hidden_size])

                # update c_g and g
                c_g_n = tf.reduce_sum(reshaped_g_f_n * c_h, axis=1) + tf.squeeze(reshaped_g_g_n, axis=1) * c_g
                g_n = g_o_n * tf.nn.tanh(c_g_n)

                '''Update h'''
                # get states before/after
                h_before = [tf.reshape(get_hidden_states_before(h, step + 1, padding), [-1, hidden_size]) for step in range(config.context_window)]
                h_before = sum_together(h_before)
                h_after = [tf.reshape(get_hidden_states_after(h, step + 1, padding), [-1, hidden_size]) for step in range(config.context_window)]
                h_after = sum_together(h_after)

                # get cells before/after
                c_h_before = [tf.reshape(get_hidden_states_before(c_h, step + 1, padding), [-1, hidden_size]) for step in range(config.context_window)]
                c_h_before = sum_together(c_h_before)
                c_h_after = [tf.reshape(get_hidden_states_after(c_h, step + 1, padding), [-1, hidden_size]) for step in range(config.context_window)]
                c_h_after = sum_together(c_h_after)

                # reshape for matmul
                reshaped_h = tf.reshape(h, [-1, hidden_size])
                reshaped_c_h = tf.reshape(c_h, [-1, hidden_size])
                reshaped_g = tf.reshape(tf.tile(tf.expand_dims(g, axis=1), [1, shape[1], 1]), [-1, hidden_size])
                reshaped_c_g = tf.reshape(tf.tile(tf.expand_dims(c_g, axis=1), [1, shape[1], 1]), [-1, hidden_size])

                # concat before and after hidden states
                h_before_after = tf.concat([h_before, h_after], axis=1)

                # forget gates for h
                f_n = tf.nn.sigmoid(tf.matmul(reshaped_h, Uf) + tf.matmul(h_before_after, Wlrf) + tf.matmul(reshaped_h, Wf) + tf.matmul(reshaped_g, Zf) + bf)
                l_n = tf.nn.sigmoid(tf.matmul(reshaped_h, Ul) + tf.matmul(h_before_after, Wlrl) + tf.matmul(reshaped_h, Wl) + tf.matmul(reshaped_g, Zl) + bl)
                r_n = tf.nn.sigmoid(tf.matmul(reshaped_h, Ur) + tf.matmul(h_before_after, Wlrr) + tf.matmul(reshaped_h, Wr) + tf.matmul(reshaped_g, Zr) + br)
                q_n = tf.nn.sigmoid(tf.matmul(reshaped_h, Uq) + tf.matmul(h_before_after, Wlrq) + tf.matmul(reshaped_h, Wq) + tf.matmul(reshaped_g, Zq) + bq)
                i_n = tf.nn.sigmoid(tf.matmul(reshaped_h, Ui) + tf.matmul(h_before_after, Wlri) + tf.matmul(reshaped_h, Wi) + tf.matmul(reshaped_g, Zi) + bi)
                o_n = tf.nn.sigmoid(tf.matmul(reshaped_h, Uo) + tf.matmul(h_before_after, Wlro) + tf.matmul(reshaped_h, Wi) + tf.matmul(reshaped_g, Zo) + bo)

                c_h_n = (l_n * c_h_before) + (r_n * c_h_after) + (f_n * reshaped_c_h) + (q_n * reshaped_c_g) + (i_n * reshaped_c_h)
                h_n = o_n * tf.nn.tanh(c_h_n)

                # update states
                h = tf.reshape(h_n, [-1, shape[1], hidden_size])
                c_h = tf.reshape(c_h_n, [-1, shape[1], hidden_size])

                g = g_n
                c_g = c_g_n

                final_hidden_states.append(h)
                final_cell_states.append(c_h)
                final_global_states.append(g)

                h = tf.nn.dropout(h, config.keep_prob)
                c_h = tf.nn.dropout(c_h, config.keep_prob)

    return final_hidden_states, final_cell_states, final_global_states

def xyz_resize(prediction, config):
    nframes = config.batch_size
    prediction = tf.reshape(prediction,[nframes,-1,3])
    njoints = prediction.get_shape().as_list()[1]
    xyz_resize = []
    index = config.chain_idx

    for n in range(nframes):
        bone_resize = []
        new_xyz = []
        bone_length = []
        for k in range(len(index)):
            for i in range(len(index[k])):
                if i == 0:
                    bone_resize.append(tf.constant([0., 0., 0.]))
                    bone_length.append(tf.constant(0.0))
                else:
                    bone = prediction[n,index[k][i],:] - prediction[n,index[k][i-1],:]
                    bone_hat = bone/tf.norm(bone)
                    new_bone = bone_hat*config.bone[index[k][i]][0]
                    bone_resize.append(new_bone)
                    bone_length.append(tf.norm(new_bone))

        for k in range(len(index)):
            for i in range(len(index[k])):
                if i == 0:
                    if k<3:
                        new_xyz.append(tf.constant([0., 0., 0.]))
                    else:
                        new_xyz.append(new_xyz[14])
                else:
                    new_xyz.append(new_xyz[index[k][i-1]] + bone_resize[index[k][i]])
        xyz_resize.append(new_xyz)

    xyz_resize = tf.stack(xyz_resize)
    xyz_resize = tf.reshape(xyz_resize,[nframes,-1])
    return xyz_resize
