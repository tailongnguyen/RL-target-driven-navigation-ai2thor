import tensorflow as tf
import numpy as np

def mse(predicted, ground_truth):
    # Mean-squared error
    return tf.square(predicted - ground_truth) / 2.

def noise_and_argmax(logits):
    logits = np.asarray(logits, dtype = np.float32)
    # Add noise then take the argmax
    noise = np.random.uniform(0, 1, logits.shape)
    
    return np.argmax(logits - np.log(-np.log(noise)))

def openai_entropy(logits):
    # Entropy proposed by OpenAI in their A2C baseline
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

class LearningRateDecay(object):
    def __init__(self, v, nvalues, lr_decay_method):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues

        def constant(p):
            return 1

        def linear(p):
            return 1 - p

        lr_decay_methods = {
            'linear': linear,
            'constant': constant
        }

        self.decay = lr_decay_methods[lr_decay_method]

    def value(self):
        current_value = self.v * self.decay(self.n / self.nvalues)
        self.n += 1.
        return current_value

    def get_value_for_steps(self, steps):
        return self.v * self.decay(steps / self.nvalues)