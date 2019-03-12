import tensorflow as tf
import numpy as np
HIDDEN_SIZE = 128

def _fc_weight_variable(shape, name='W_fc'):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.get_variable(name=name, dtype = tf.float32, initializer=initial)

def _fc_bias_variable(shape, input_channels, name='b_fc'):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.get_variable(name=name, dtype=tf.float32, initializer=initial)


class QNetwork():
    def __init__(self, name, state_size, action_size, history_size=1, dropout_keep_prob=-1):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, history_size, self.state_size])

            self.inputs_flat = tf.reshape(self.inputs, [-1, self.state_size * history_size])
            self.actions = tf.placeholder(tf.float32, [None, self.action_size])
            self.target_Q = tf.placeholder(tf.float32, [None])
            self.learning_rate = tf.placeholder(tf.float32, [])

            self.W_fc1 = _fc_weight_variable([self.state_size * history_size, HIDDEN_SIZE], name="W_fc1")
            self.b_fc1 = _fc_bias_variable([HIDDEN_SIZE], self.state_size, name="b_fc1")
            self.fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.inputs_flat, self.W_fc1), self.b_fc1))

            if dropout_keep_prob != -1:
                self.fc1 = tf.nn.dropout(self.fc1, dropout_keep_prob)

            self.W_fc2 = _fc_weight_variable([HIDDEN_SIZE, self.action_size], name="W_fc2")
            self.b_fc2 = _fc_bias_variable([self.action_size], HIDDEN_SIZE, name="b_fc2")

            self.q_values = tf.nn.bias_add(tf.matmul(self.fc1, self.W_fc2), self.b_fc2)
            self.Q_expected = tf.reduce_sum(tf.multiply(self.q_values, self.actions))


            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q_expected))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
