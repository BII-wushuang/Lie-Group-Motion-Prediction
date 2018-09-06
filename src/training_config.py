import numpy as np
import tensorflow as tf

class train_Config(object):

    """Training Configurations"""
    max_grad_norm = 5                           # Gradient clipping threshold
    input_window_size = 50                      # Input window size during training
    output_window_size = 10                     # Output window size during training
    test_output_window = 100                    # Output window size during testing. test_output_window is overwritten by test set size when longterm is true
    hidden_size = 300                           # Number of hidden units for HMR
    batch_size = 8                              # Batch size for training
    learning_rate = 0.001                       # Learning rate
    max_epoch = 50                              # Maximum training epochs
    training_size = 1000                        # Training iterations per epoch
    validation_size = 50                        # Validation iterations per epoch
    restore = False                             # Restore the trained weights or restart training from scratch
    longterm = False                            # Whether we are doing super longterm prediction
    early_stop = 10                             # Stop training if validation loss do not improve after these epochs
    keep_prob = 1                               # Keep probability for RNN cell weights
    context_window = 1                          # Context window size in HMR
    recurrent_steps = 10                        # Number of recurrent steps in HMR

    """Choice of model and loss function"""
    models = ['ERD', 'LSTM3lr', 'GRU', 'HMR']
    model = models[3]

    loss_funcs = ['l2', 'linearizedlie', 'lie', 'angle']
    loss = loss_funcs[0]

    def __init__(self, dataset, datatype, action):
        self.dataset = dataset
        self.datatype = datatype
        self.filename = action

        """Define kinematic chain configurations based on dataset class"""
        if self.dataset == 'Fish':
            self.filename = 'default'
            self.chain_config = np.array([21])
        elif self.dataset == 'Mouse':
            self.filename = 'default'
            self.chain_config = np.array([5])
        else: # Human
            self.chain_config = np.array([6, 6, 5, 5, 5])

        """Setting kinematic chains configurations, do not modify the code below"""
        self.nchains = self.chain_config.shape[0]
        self.skip = np.zeros([self.nchains])

        for i in range(self.nchains):
            if i == 0:
                self.skip[i] = self.chain_config[i]
            else:
                self.skip[i] = self.skip[i - 1] + self.chain_config[i]

        self.skip = np.concatenate((np.array([0]), self.skip))
        self.skip = self.skip.astype(int)

        self.chain_idx = []
        for j in range(self.skip.shape[0] - 1):
            self.chain_idx.append(np.arange(self.skip[j], self.skip[j + 1]))

        self.idx = [0]
        for j in range(len(self.chain_idx)):
            self.idx.append(self.chain_idx[j][-1] - j)