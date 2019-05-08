"""
This code is same from https://github.com/siddk/npi
npi.py
Core model definition script for the Neural Programmer-Interpreter.
"""
import tensorflow as tf
import tflearn


class NPI():
    def __init__(self, core, config, npi_core_dim=256, npi_core_layers=2, verbose=0):
        """
        Instantiate an NPI Model, with the necessary hyperparameters, including the task-specific
        core.

        :param core: Task-Specific Core, with fields representing the environment state vector,
                     the input placeholders, and the program embedding.
        :param config: Task-Specific Configuration Dictionary, with fields representing the
                       necessary parameters.
        """
        self.core, self.state_dim, self.program_dim = core, core.state_dim, core.program_dim
        self.bsz, self.npi_core_dim, self.npi_core_layers = core.bsz, npi_core_dim, npi_core_layers
        self.env_in, self.prg_in = core.env_in,  core.prg_in
        self.state_encoding, self.program_embedding = core.state_encoding, core.program_embedding
        self.num_progs, self.key_dim = config["PROGRAM_NUM"], config["PROGRAM_KEY_SIZE"]

        # Setup Label Placeholders
        self.y_term = tf.placeholder(tf.int64, shape=[None], name='Termination_Y')
        self.y_prog = tf.placeholder(tf.int64, shape=[None], name='Program_Y')

        # Build NPI LSTM Core, hidden state
        self.reset_state()
        self.h = self.npi_core()

        # Build Termination Network => Returns probability of terminating
        self.terminate = self.terminate_net()

        # Build Key Network => Generates probability distribution over programs
        self.program_distribution = self.key_net()

        # Build Argument Networks => Generates list of argument distributions

        # Build Losses
        self.t_loss, self.p_loss = self.build_losses()
        self.default_loss = 2 * self.t_loss + self.p_loss

        # Build Optimizer
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.0001, self.global_step, 10000, 0.95,
                                                        staircase=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Build Metrics
        self.t_metric, self.p_metric = self.build_metrics()
        self.metrics = [self.t_metric, self.p_metric]

        # Build Train Ops
        self.default_train_op = self.opt.minimize(self.default_loss, global_step=self.global_step)

    def reset_state(self):
        """
        Zero NPI Core LSTM Hidden States. LSTM States are represented as a Tuple, consisting of the
        LSTM C State, and the LSTM H State (in that order: (c, h)).
        """
        zero_state = tf.zeros([self.bsz, 2 * self.npi_core_dim])
        self.h_states = [zero_state for _ in range(self.npi_core_layers)]

    def npi_core(self):
        """
        Build the NPI LSTM core, feeding the program embedding and state encoding to a multi-layered
        LSTM, returning the h-state of the final LSTM layer.

        References: Reed, de Freitas [2]
        """
        s_in = self.state_encoding                               # Shape: [bsz, state_dim]
        p_in = self.program_embedding                            # Shape: [bsz, 1, program_dim]

        # Reshape state_in
        s_in = tflearn.reshape(s_in, [-1, 1, self.state_dim])    # Shape: [bsz, 1, state_dim]

        # Concatenate s_in, p_in
        c = tflearn.merge([s_in, p_in], 'concat', axis=2)        # Shape: [bsz, 1, state + prog]

        # Feed through Multi-Layer LSTM
        # print('-'*100)
        net, state = tflearn.layers.recurrent.lstm(c, self.npi_core_dim, return_seq=True, return_state=True)
        net, state = tflearn.layers.recurrent.lstm(net, self.npi_core_dim, return_seq=True, return_state=True)
        # print('*'*100)
        # print(state)
        top_state = state.c
        return top_state

    def terminate_net(self):
        """
        Build the NPI Termination Network, that takes in the NPI Core Hidden State, and returns
        the probability of terminating program.

        References: Reed, de Freitas [3]
        """
        p_terminate = tflearn.fully_connected(self.h, 2, activation='linear', regularizer='L2')
        return p_terminate                                      # Shape: [bsz, 2]

    def key_net(self):
        """
        Build the NPI Key Network, that takes in the NPI Core Hidden State, and returns a softmax
        distribution over possible next programs.

        References: Reed, de Freitas [3, 4]
        """
        # Get Key from Key Network
        hidden = tflearn.fully_connected(self.h, self.key_dim, activation='elu', regularizer='L2')
        key = tflearn.fully_connected(hidden, self.key_dim)    # Shape: [bsz, key_dim]

        # Perform dot product operation, then softmax over all options to generate distribution
        key = tflearn.reshape(key, [-1, 1, self.key_dim])
        key = tf.tile(key, [1, self.num_progs, 1])             # Shape: [bsz, n_progs, key_dim]
        prog_sim = tf.multiply(key, self.core.program_key)          # Shape: [bsz, n_progs, key_dim]
        prog_dist = tf.reduce_sum(prog_sim, [2])               # Shape: [bsz, n_progs]
        return prog_dist

    def build_losses(self):
        """
        Build separate loss computations, using the logits from each of the sub-networks.
        """
        # Termination Network Loss
        termination_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.terminate, labels=self.y_term), name='Termination_Network_Loss')

        # Program Network Loss
        program_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.program_distribution, labels=self.y_prog), name='Program_Network_Loss')

        return termination_loss, program_loss#, arg_losses

    def build_metrics(self):
        """
        Build accuracy metrics for each of the sub-networks.
        """
        term_metric = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.terminate, 1), self.y_term),  tf.float32), name='Termination_Accuracy')

        program_metric = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.program_distribution, 1), self.y_prog), tf.float32), name='Program_Accuracy')
        return term_metric, program_metric
