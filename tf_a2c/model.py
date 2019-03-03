import numpy as np
import tensorflow as tf
import os

from utils import openai_entropy, mse, LearningRateDecay

HIDDEN_SIZE = 512

def _fc_weight_variable(shape, name='W_fc'):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.get_variable(name=name, dtype = tf.float32, initializer=initial)

def _fc_bias_variable(shape, input_channels, name='b_fc'):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.get_variable(name=name, dtype=tf.float32, initializer=initial)

class Actor():
    def __init__(self, state_size, action_size, history_size=1, dropout_keep_prob=-1, embedding_size=-1, reuse=False):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope('Actor' if not reuse else "ShareLatent"):
            self.inputs = tf.placeholder(tf.float32, [None, history_size, self.state_size])
            self.inputs_flat = tf.reshape(self.inputs, [-1, self.state_size * history_size])

            if embedding_size != -1:
                self.task_input = tf.placeholder(tf.float32, [None, embedding_size])
                self.inputs_flat = tf.concat([self.task_input, self.inputs_flat], 1)

            self.actions = tf.placeholder(tf.int32, [None, self.action_size])
            self.advantages = tf.placeholder(tf.float32, [None, ])

            if embedding_size != -1:
                self.W_fc1 = _fc_weight_variable([self.state_size * history_size + embedding_size, HIDDEN_SIZE], name = "W_fc1")
            else:
                self.W_fc1 = _fc_weight_variable([self.state_size * history_size, HIDDEN_SIZE], name = "W_fc1")

            self.b_fc1 = _fc_bias_variable([HIDDEN_SIZE], self.state_size, name = "b_fc1")
            self.fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.inputs_flat, self.W_fc1), self.b_fc1))

            if dropout_keep_prob != -1:
                self.fc1 = tf.nn.dropout(self.fc1, dropout_keep_prob)

        with tf.variable_scope("Actions"):
            self.W_fc2 = _fc_weight_variable([HIDDEN_SIZE, self.action_size], name = "W_fc2")
            self.b_fc2 = _fc_bias_variable([self.action_size], HIDDEN_SIZE, name = "b_fc2")

        self.logits = tf.nn.bias_add(tf.matmul(self.fc1, self.W_fc2), self.b_fc2)

        self.pi = tf.nn.softmax(self.logits)
        self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.actions)
        self.policy_loss = tf.reduce_mean(self.neg_log_prob * self.advantages)

        self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

class Critic():
    def __init__(self, state_size, history_size=1, dropout_keep_prob=-1, embedding_size=-1, reuse=False):
        self.state_size = state_size

        with tf.variable_scope('Critic' if not reuse else "ShareLatent" , reuse=reuse):
            self.inputs = tf.placeholder(tf.float32, [None, history_size, self.state_size])
            self.returns = tf.placeholder(tf.float32, [None, ])

            self.inputs_flat = tf.reshape(self.inputs, [-1, self.state_size * history_size])

            if embedding_size != -1:
                self.task_input = tf.placeholder(tf.float32, [None, embedding_size])
                self.inputs_flat = tf.concat([self.task_input, self.inputs_flat], 1)
                self.W_fc1 = _fc_weight_variable([self.state_size * history_size + embedding_size, HIDDEN_SIZE], name = "W_fc1")
            else:
                self.W_fc1 = _fc_weight_variable([self.state_size * history_size, HIDDEN_SIZE], name = "W_fc1")

            self.b_fc1 = _fc_bias_variable([HIDDEN_SIZE], self.state_size, name = "b_fc1")
            self.fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.inputs_flat, self.W_fc1), self.b_fc1))

            if dropout_keep_prob != -1:
                self.fc1 = tf.nn.dropout(self.fc1, dropout_keep_prob)

        with tf.variable_scope("Value"):
            self.W_fc2 = _fc_weight_variable([HIDDEN_SIZE, 1], name = "W_fc3")
            self.b_fc2 = _fc_bias_variable([1], HIDDEN_SIZE, name = "b_fc3")

            self.value = tf.nn.bias_add(tf.matmul(self.fc1, self.W_fc2), self.b_fc2)
            
        self.value_loss = tf.reduce_mean(mse(tf.squeeze(self.value), self.returns))
   
        self.variables = [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

class A2C():
    def __init__(self, 
                name, 
                state_size, 
                action_size, 
                history_size,
                embedding_size,
                entropy_coeff, 
                value_function_coeff, 
                max_gradient_norm=None, 
                dropout=-1,
                joint_loss=False, 
                learning_rate=None,
                alpha=0.97,
                epsilon=1e-5,
                decay=False, 
                reuse=False):

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
        self.success_rate = tf.placeholder(tf.float32)
        
        with tf.variable_scope(name):
            self.actor = Actor(state_size=self.state_size, action_size=self.action_size, 
                            history_size=history_size, dropout_keep_prob=dropout,
                            embedding_size=embedding_size, reuse=self.reuse)
            self.critic = Critic(state_size=self.state_size, history_size=history_size,
                            embedding_size=embedding_size, dropout_keep_prob=dropout, reuse=self.reuse)

        self.learning_rate = tf.placeholder(tf.float32, [])
        self.fixed_lr = learning_rate
        self.decay = decay

        if self.joint_loss:
            self.entropy = tf.reduce_mean(openai_entropy(self.actor.logits))
            self.total_loss = self.actor.policy_loss + self.critic.value_loss * self.value_function_coeff - self.entropy * self.entropy_coeff

            with tf.variable_scope(name + '/joint_opt'):
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=alpha, epsilon=epsilon)
                params = self.actor.variables + self.critic.variables
                grads = tf.gradients(self.total_loss, params)

                if self.max_gradient_norm is not None:
                    grads, grad_norm = tf.clip_by_global_norm(grads, max_gradient_norm)
                    grads = list(zip(grads, params))

                    self.train_opt_joint = optimizer.apply_gradients(grads)
                else:
                    self.train_opt_joint = optimizer.minimize(self.total_loss)
        else:

            with tf.variable_scope(name + '/actor_opt'):
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=alpha, epsilon=epsilon)
                params = self.actor.variables
                grads = tf.gradients(self.actor.policy_loss, params)

                if self.max_gradient_norm is not None:
                    grads, grad_norm = tf.clip_by_global_norm(grads, max_gradient_norm)
                    grads = list(zip(grads, params))

                    self.train_opt_policy = optimizer.apply_gradients(grads)
                else:
                    self.train_opt_policy = optimizer.minimize(self.actor.policy_loss)


            with tf.variable_scope(name + '/critic_opt'):
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=alpha, epsilon=epsilon)
                params = self.critic.variables
                grads = tf.gradients(self.critic.value_loss, params)

                if self.max_gradient_norm is not None:
                    grads, grad_norm = tf.clip_by_global_norm(grads, max_gradient_norm)
                    grads = list(zip(grads, params))

                    self.train_opt_value = optimizer.apply_gradients(grads)
                else:
                    self.train_opt_value = optimizer.minimize(self.critic.value_loss)

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
        
    def learn(self, sess, actor_states, critic_states, actions, returns, advantages, task_inputs=[]):
        if self.decay:
            for i in range(len(actor_states)):
                current_learning_rate = self.learning_rate_decayed.value()
        else:
            current_learning_rate = self.fixed_lr

        if len(task_inputs) == 0:
            feed_dict = {
                            self.actor.inputs: actor_states, 
                            self.critic.inputs: critic_states, 
                            self.critic.returns: returns,
                            self.actor.actions: actions, 
                            self.actor.advantages: advantages,
                            self.learning_rate: current_learning_rate,
                        }
        else:
            feed_dict = {
                            self.actor.inputs: actor_states, 
                            self.actor.task_input: task_inputs,
                            self.critic.inputs: critic_states, 
                            self.critic.returns: returns,
                            self.critic.task_input: task_inputs,
                            self.actor.actions: actions, 
                            self.actor.advantages: advantages,
                            self.learning_rate: current_learning_rate,
                        }

        if self.joint_loss:
            try:
                policy_loss, value_loss, policy_entropy, total_loss, _ = sess.run(
                    [self.actor.policy_loss, self.critic.value_loss, self.entropy, self.total_loss, self.train_opt_joint],
                    feed_dict = feed_dict
                )
            except ValueError:
                import sys
                print("Actor states: ", actor_states)
                print("Returns: ", returns)
                print("Actions: ", actions)
                print("Advantages: ", advantages)
                sys.exit()

            return policy_loss, value_loss, policy_entropy, total_loss
        else:
            policy_loss, value_loss, _, _ = sess.run(
                [self.actor.policy_loss, self.critic.value_loss, self.train_opt_policy, self.train_opt_value], 
                feed_dict = feed_dict)

            return policy_loss, value_loss, None, None

        
if __name__ == '__main__':
    a2c = A2C(100, 8, 0.05, 0.5, reuse = True)