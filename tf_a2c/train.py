import tensorflow as tf
import numpy as np      			
import os
import sys 

from model import *
from rollout import Rollout
from env.ai2thor_env import AI2ThorDumpEnv

class MultitaskPolicy(object):

	def __init__(
			self,
			training_scene,
			training_objects,
			config,
			custom_config
			):

		self.config = config
		self.custom_config = custom_config

		self.use_gae = custom_config.get('use_gae') or config['use_gae']
		self.num_task = custom_config.get('num_task') or config['num_task']
		self.num_epochs = custom_config.get('num_epochs') or config['num_epochs']
		self.num_episodes = custom_config.get('num_episodes') or config['num_episodes']
		self.num_iters = custom_config.get('num_iters') or config['num_iters']
		self.gamma = custom_config.get('gamma') or config['gamma']
		self.lamb = custom_config.get('lamb') or config['lamb']
		self.lr = custom_config.get('lr') or config['lr']
		self.joint_loss = custom_config.get('joint_loss') or config['joint_loss']
		self.ec = custom_config.get('ec') or config['ec']
		self.vc = custom_config.get('vc') or config['vc']
		self.max_grad_norm = custom_config.get('max_gradient_norm') or config['max_gradient_norm']
		self.decay = custom_config.get('decay') or config['decay']
		self.reuse = custom_config.get('share_latent') or config['share_latent']

		assert len(training_objects) >= self.num_task, "Each task should have an unique target."

		self.env = AI2ThorDumpEnv(training_scene, training_objects[0], config, custom_config)
		self.rollout = Rollout(training_scene, training_objects, config, custom_config)

		tf.reset_default_graph()

		self.PGNetwork = []
		for i in range(self.num_task):
			policy_i = A2C(name='A2C_' + str(i), 
							state_size=self.env.features.shape[1], 
							action_size=self.env.action_space,
							entropy_coeff=self.ec,
							value_function_coeff=self.vc,
							max_gradient_norm=self.max_grad_norm,
							joint_loss=self.joint_loss,
							learning_rate=self.lr,
							decay=self.decay,
							reuse=self.reuse
							)

			if self.decay:
				policy_i.set_lr_decay(self.lr, self.num_epochs * self.num_episodes * self.num_iters)
			
			print("\nInitialized network {}, with {} trainable weights.".format('A2C_' + str(i), len(policy_i.find_trainable_variables('A2C_' + str(i), True))))
			self.PGNetwork.append(policy_i)

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.2)

		self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
		self.sess.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver()
		
		log_folder = custom_config.get('logging') or config['logging']
		self.writer = tf.summary.FileWriter(log_folder)
		
		test_name =  "{}_{}".format(training_scene, "_".join(training_objects[:self.num_task]))
		tf.summary.scalar(test_name + "/rewards", tf.reduce_mean([policy.mean_reward for policy in self.PGNetwork], 0))

		self.write_op = tf.summary.merge_all()

	def prepare_current(self, sess):
		current_policy = {}
		current_values = {}
		for task in range(self.num_task):
			for state_index in range(self.env.features.shape[0]):
				logit, p = sess.run(
							[self.PGNetwork[task].actor.logits, self.PGNetwork[task].actor.pi], 
							feed_dict={
								self.PGNetwork[task].actor.inputs: [np.squeeze(self.env.features[state_index], -1)],
							})
		
				current_policy[state_index, task, 0] = logit.ravel().tolist()
				current_policy[state_index, task, 1] = p.ravel().tolist()
				
				v = sess.run(
							self.PGNetwork[task].critic.value, 
							feed_dict={
								self.PGNetwork[task].critic.inputs: [np.squeeze(self.env.features[state_index], -1)],
							})
			
				current_values[state_index, task] = v.ravel().tolist()[0]

		return current_policy, current_values

	def discount_with_dones(self, rewards, dones, gamma):
		discounted = []
		r = 0
		# Start from downwards to upwards like Bellman backup operation.
		for reward, done in zip(rewards[::-1], dones[::-1]):
			r = reward + gamma * r * (1. - done)  # fixed off by one bug
			discounted.append(r)
		return discounted[::-1]

	def generalized_advantage_estimate(self, rewards, dones, values, last_value, gamma, lamb):
		advantages = np.zeros_like(rewards)
		lastgaelam = 0

        # From last step to first step
		for t in reversed(range(len(rewards))):
            # If t == before last step
			if t == len(rewards) - 1:
				# If a state is done, nextnonterminal = 0
				# In fact nextnonterminal allows us to do that logic

				#if done (so nextnonterminal = 0):
				#    delta = R - V(s) (because self.gamma * nextvalues * nextnonterminal = 0) 
				# else (not done)
				    #delta = R + gamma * V(st+1)
				nextnonterminal = 1.0 - dones[-1]

				# V(t+1)
				nextvalue = last_value
			else:
				nextnonterminal = 1.0 - dones[t]

				nextvalue = values[t+1]

			# Delta = R(t) + gamma * V(t+1) * nextnonterminal  - V(t)
			delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]

			# Advantage = delta + gamma *  (lambda) * nextnonterminal  * lastgaelam
			advantages[t] = lastgaelam = delta + gamma * lamb * nextnonterminal * lastgaelam

		# advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
		return list(advantages)

	def _make_batch(self, sess):



		current_policy, current_values = self.prepare_current(sess)

		'''
		states = [
		    task1		[[---episode_1---],...,[---episode_n---]],
		    task2		[[---episode_1---],...,[---episode_n---]],
		   .
		   .
			task_k      [[---episode_1---],...,[---episode_n---]],
		]
		same as actions, tasks, rewards, values, dones
		
		last_values = [
			task1		[---episode_1---, ..., ---episode_n---],
		    task2		[---episode_1---, ..., ---episode_n---],
		   .
		   .
			task_k      [---episode_1---, ..., ---episode_n---],	
		]
		'''
		states, tasks, actions, rewards, next_states = self.rollout.rollout_batch(current_policy)

		observations = [[] for i in range(self.num_task)]
		converted_actions = [[] for i in range(self.num_task)]
		task_logits = [[] for i in range(self.num_task)]

		for task_idx, task_states in enumerate(states):
			for ep_idx, ep_states in enumerate(task_states):
				observations[task_idx] += [np.squeeze(self.env.features[si], -1) for si in ep_states]
				converted_actions[task_idx] += [self.env.cv_action_onehot[a] for a in actions[task_idx][ep_idx]]
				task_logits[task_idx] += [current_policy[si, task_idx, 0] for si in ep_states]

		returns = [[] for i in range(self.num_task)]
		advantages = [[] for i in range(self.num_task)]

		if not self.use_gae:

			for task_idx in range(self.num_task):
				for ep_idx, (ep_rewards, ep_states, ep_next_states) in enumerate(zip(rewards[task_idx], states[task_idx], next_states[task_idx])):
					assert len(ep_rewards) == len(ep_states) == len(ep_next_states)
					ep_dones = list(np.zeros_like(ep_rewards))

					if ep_rewards[-1] != self.config['success_reward']:
						last_value = current_values[ep_next_states[-1], task_idx]
						ep_returns = self.discount_with_dones(ep_rewards + [last_value], ep_dones+[0], self.gamma)[:-1]
					else:
						ep_dones[-1] = 1
						ep_returns = self.discount_with_dones(ep_rewards, ep_dones, self.gamma)

					returns[task_idx] += ep_returns
					ep_values = [current_values[s, task_idx] for s in ep_states]

					# Here we calculate advantage A(s,a) = R + yV(s') - V(s)
			    	# rewards = R + yV(s')
					advantages[task_idx] += list((np.array(ep_returns) - np.array(ep_values)).astype(np.float32))

		else:

			for task_idx in range(self.num_task):
				for ep_idx, (ep_rewards, ep_states, ep_next_states) in enumerate(zip(rewards[task_idx], states[task_idx], next_states[task_idx])):
					ep_dones = list(np.zeros_like(ep_rewards))

					# print(ep_rewards)
					if ep_rewards[-1] != self.config['success_reward']:
						last_value = current_values[ep_next_states[-1], task_idx]
						returns[task_idx] += self.discount_with_dones(ep_rewards + [last_value], ep_dones+[0], self.gamma)[:-1]
					else:
						last_value = 0
						ep_dones[-1] = 1
						returns[task_idx] += self.discount_with_dones(ep_rewards, ep_dones, self.gamma)

					ep_values = [current_values[s, task_idx] for s in ep_states]
					advantages[task_idx] += self.generalized_advantage_estimate(ep_rewards, ep_dones, ep_values, last_value, self.gamma, self.lamb)

					# returns[task_idx] += self._discount_rewards(ep_rewards, ep_next_states, task_idx, current_values)
					# advantages[task_idx] += self._GAE(ep_rewards, ep_states, ep_next_states, task_idx, current_values)
					
				assert len(returns[task_idx]) == len(advantages[task_idx])
		
		return observations,\
				 converted_actions,\
				 returns,\
				 advantages,\
				 task_logits,\
				 rewards				 		
		
	def train(self):
		total_samples = {}

		for epoch in range(self.num_epochs):
			# sys.stdout.flush()
			
			# ROLLOUT SAMPLE
			#---------------------------------------------------------------------------------------------------------------------#	
			mb_states,\
			mb_actions,\
			mb_returns,\
			mb_advantages,\
			mb_logits,\
			rewards = self._make_batch(self.sess)
			#---------------------------------------------------------------------------------------------------------------------#	

			print('epoch {}/{}'.format(epoch + 1, self.num_epochs), end = '\r', flush = True)
		
			sum_dict = {}
			for task_idx in range(self.num_task):

				assert len(mb_states[task_idx]) == len(mb_actions[task_idx]) == len(mb_returns[task_idx]) == len(mb_advantages[task_idx])

				policy_loss, value_loss, _, _ = self.PGNetwork[task_idx].learn(self.sess, 
																				actor_states = mb_states[task_idx],
																				advantages = mb_advantages[task_idx],
																				actions = mb_actions[task_idx],
																				critic_states = mb_states[task_idx],
																				returns = mb_returns[task_idx],
																				task_logits = mb_logits[task_idx]
																			)


				sum_dict[self.PGNetwork[task_idx].mean_reward] = np.sum(np.concatenate(rewards[task_idx])) / len(rewards[task_idx])

				if task_idx not in total_samples:
					total_samples[task_idx] = 0
					
				total_samples[task_idx] += len(list(np.concatenate(rewards[task_idx])))

			#---------------------------------------------------------------------------------------------------------------------#	
			

			# WRITE TF SUMMARIES
			#---------------------------------------------------------------------------------------------------------------------#	
			summary = self.sess.run(self.write_op, feed_dict = sum_dict)

			self.writer.add_summary(summary, np.mean(list(total_samples.values())))
			self.writer.flush()
			#---------------------------------------------------------------------------------------------------------------------#	

		self.sess.close()
		# SAVE MODEL
		#---------------------------------------------------------------------------------------------------------------------#	
		
			
		#---------------------------------------------------------------------------------------------------------------------#		