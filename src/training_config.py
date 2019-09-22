import numpy as np
import tensorflow as tf


class train_Config(object):

    """Training Configurations"""
    max_grad_norm = 5                           # Gradient clipping threshold
    input_window_size = 50                      # Input window size during training
    output_window_size = 10                     # Output window size during training
    test_output_window = 100                    # Output window size during testing. test_output_window is overwritten by test set size when longterm is true
    hidden_size = 100                           # Number of hidden units for HMR
    batch_size = 8                              # Batch size for training
    learning_rate = 0.001                       # Learning rate
    max_epoch = 50                              # Maximum training epochs
    training_size = 500                         # Training iterations per epoch
    validation_size = 50                        # Validation iterations per epoch
    restore = False                             # Restore the trained weights or restart training from scratch
    longterm = False                            # Whether we are doing super longterm prediction
    early_stop = 10                             # Stop training if validation loss do not improve after these epochs
    keep_prob = 0.9                             # Keep probability for RNN cell weights
    context_window = 1                          # Context window size in HMR
    recurrent_steps = 3                         # Number of recurrent steps in HMR

    """Choice of model and loss function"""
    models = ['ERD', 'LSTM3lr', 'GRU', 'HMR']
    model = models[3]

    loss_funcs = ['l2', 'linearizedlie']
    loss = loss_funcs[0]

    def __init__(self, dataset, datatype, action):
        self.dataset = dataset
        self.datatype = datatype
        self.filename = action

        """Define kinematic chain configurations based on dataset class"""
        if self.dataset == 'Fish':
            self.filename = 'default'
            self.chain_config = [np.arange(0, 21)]
        elif self.dataset == 'Mouse':
            self.filename = 'default'
            self.chain_config = [np.arange(0, 5)]
        elif self.dataset == 'Human':
            self.chain_config = [np.array([0, 1, 2, 3, 4, 5]),
                                 np.array([0, 6, 7, 8, 9, 10]),
                                 np.array([0, 12, 13, 14, 15]),
                                 np.array([13, 17, 18, 19, 22, 19, 21]),
                                 np.array([13, 25, 26, 27, 30, 27, 29])]
