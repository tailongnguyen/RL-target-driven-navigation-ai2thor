import numpy as np
import tensorflow as tf
import os

from utils import openai_entropy, mse, LearningRateDecay

HIDDEN_SIZE = 512
class Actor():
    def __init__(self, state_size, action_size, reuse = False):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope('Actor' if not reuse else "ShareLatent"):
            self.inputs = tf.placeholder(tf.float32, [None, self.state_size])
            self.actions = tf.placeholder(tf.int32, [None, self.action_size])
            self.advantages = tf.placeholder(tf.float32, [None, ])

            self.W_fc1 = self._fc_weight_variable([self.state_size, 512], name = "W_fc1")
            self.b_fc1 = self._fc_bias_variable([512], self.state_size, name = "b_fc1")
            self.fc1 = tf.nn.relu(tf.matmul(self.inputs, self.W_fc1) + self.b_fc1)

        with tf.variable_scope("Actions"):
            self.W_fc2 = self._fc_weight_variable([512, self.action_size], name = "W_fc2")
            self.b_fc2 = self._fc_bias_variable([self.action_size], 512, name = "b_fc2")

        self.logits = tf.matmul(self.fc1, self.W_fc2) + self.b_fc2

        self.pi = tf.nn.softmax(self.logits)
        self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.actions)
        self.policy_loss = tf.reduce_mean(self.neg_log_prob * self.advantages)

        self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

    def _fc_weight_variable(self, shape, name='W_fc'):
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

    def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)


class Critic():
    def __init__(self, state_size, reuse = False):
        self.state_size = state_size

        with tf.variable_scope('Critic' if not reuse else "ShareLatent" , reuse  = reuse):
            self.inputs = tf.placeholder(tf.float32, [None, self.state_size])
            self.returns = tf.placeholder(tf.float32, [None, ])

            self.W_fc1 = self._fc_weight_variable([self.state_size, 512], name = "W_fc1")
            self.b_fc1 = self._fc_bias_variable([512], self.state_size, name = "b_fc1")
            self.fc1 = tf.nn.relu(tf.matmul(self.inputs, self.W_fc1) + self.b_fc1)

        with tf.variable_scope("Value", reuse = False):
            self.W_fc2 = self._fc_weight_variable([512, 1], name = "W_fc3")
            self.b_fc2 = self._fc_bias_variable([1], 512, name = "b_fc3")

            self.value = tf.matmul(self.fc1, self.W_fc2) + self.b_fc2
            
        self.value_loss = tf.reduce_mean(mse(tf.squeeze(self.value), self.returns))
   
        self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
            
    def _fc_weight_variable(self, shape, name='W_fc'):
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

    def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.get_variable(name=name, dtype = tf.float32, initializer = initial)

class A2C():
    def __init__(self, 
                name, 
                state_size, 
                action_size, 
                entropy_coeff, 
                value_function_coeff, 
                max_gradient_norm, 
                clip_range = 0.2,
                joint_loss = False, 
                learning_rate = None, 
                decay = False, 
                reuse = False):

        self.name = name 
        self.max_gradient_norm  = max_gradient_norm
        self.entropy_coeff = entropy_coeff
        self.value_function_coeff = value_function_coeff
        self.state_size = state_size
        self.action_size = action_size
        self.reuse = reuse
        self.joint_loss = joint_loss

        # Add this placeholder for having this variable in tensorboard
        self.mean_reward = tf.placeholder(tf.float32)
        self.mean_redundant = tf.placeholder(tf.float32)
        self.vloss_summary = tf.placeholder(tf.float32)
        self.aloss_summary = tf.placeholder(tf.float32)
        self.entropy_summary = tf.placeholder(tf.float32)
        
        with tf.variable_scope(name):
            self.actor = Actor(state_size=self.state_size, action_size=self.action_size, reuse=self.reuse)
            self.critic = Critic(state_size=self.state_size, reuse=self.reuse)

        self.learning_rate = tf.placeholder(tf.float32, [])
        self.fixed_lr = learning_rate
        self.decay = decay
        self.clip_range = clip_range

        # Calculate ratio (pi current policy / pi task policy)
        # Task logits is the placeholder for the logits of its original task
        self.task_logits = tf.placeholder(tf.float32, [None, self.action_size])
        self.task_neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.task_logits, labels=self.actor.actions)
        
        # self.ratio = tf.exp(self.task_neg_log_prob - self.actor.neg_log_prob)
        self.ratio = tf.exp(self.task_neg_log_prob - self.task_neg_log_prob)
        
        self.policy_loss1 = self.actor.policy_loss * self.ratio
        self.policy_loss2 = self.actor.policy_loss * tf.clip_by_value(self.ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        self.policy_loss = tf.reduce_mean(tf.maximum(self.policy_loss1, self.policy_loss2))

        with tf.variable_scope(name + '/actor_opt'):
            self.train_opt_policy = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.policy_loss)

        with tf.variable_scope(name + '/critic_opt'):
            self.train_opt_value = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.critic.value_loss)

        if self.joint_loss:

            self.entropy = tf.reduce_mean(openai_entropy(self.actor.logits))
            self.total_loss = self.policy_loss + self.critic.value_loss * self.value_function_coeff - self.entropy * self.entropy_coeff

            with tf.variable_scope(name + '/joint_opt'):
                self.train_opt_joint = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.total_loss)

    def set_lr_decay(self, lr_rate, nvalues):
        self.learning_rate_decayed = LearningRateDecay(v=lr_rate,
                                                       nvalues=nvalues,
                                                       lr_decay_method='linear')
        print("Learning rate decay-er has been set up!")

    def find_trainable_variables(self, key, printing = False):
        with tf.variable_scope(key):
            variables = tf.trainable_variables(key)
            if printing:
                print(len(variables), variables)
            return variables

    def save_model(self, sess, save_dir):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, self.name)
        self.saver.save(sess, save_path)

    def restore_model(self, sess, save_dir):
        save_path = os.path.join(save_dir, self.name)
        self.saver.restore(sess, save_path)
        
    def learn(self, sess, actor_states, critic_states, actions, returns, advantages, task_logits):
        if self.decay:
            for i in range(len(actor_states)):
                current_learning_rate = self.learning_rate_decayed.value()
        else:
            current_learning_rate = self.fixed_lr

        feed_dict = {
                        self.actor.inputs: actor_states, 
                        self.critic.inputs: critic_states, 
                        self.critic.returns: returns,
                        self.actor.actions: actions, 
                        self.actor.advantages: advantages,
                        self.task_logits : task_logits,
                        self.learning_rate: current_learning_rate,
                    }

        if self.joint_loss:
            try:
                policy_loss, value_loss, policy_entropy, total_loss, _ = sess.run(
                    [self.policy_loss, self.critic.value_loss, self.entropy, self.total_loss, self.train_opt_joint],
                    feed_dict = feed_dict
                )
            except ValueError:
                import sys
                print("Actor states: ", actor_states)
                print("Logits: ", task_logits)
                print("Returns: ", returns)
                print("Actions: ", actions)
                print("Advantages: ", advantages)
                sys.exit()

            return policy_loss, value_loss, policy_entropy, total_loss
        else:
            policy_loss, value_loss, _, _ = sess.run(
                [self.policy_loss, self.critic.value_loss, self.train_opt_policy, self.train_opt_value], 
                feed_dict = feed_dict)

            return policy_loss, value_loss, None, None

        
if __name__ == '__main__':
    a2c = A2C(100, 8, 0.05, 0.5, reuse = True)