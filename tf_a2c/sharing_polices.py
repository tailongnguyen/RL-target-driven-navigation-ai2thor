import tensorflow as tf
import numpy as np      			
import os
import sys 
import json
import time
import h5py
import pickle
import sys

sys.path.append('..') # to access env package

from datetime import datetime
from model import *
from rollout import Rollout
from env.ai2thor_env import AI2ThorDumpEnv

class SharingPolicy(object):

	def __init__(
			self,
			training_scene,
			training_objects,
			config,
			arguments
			):

		self.config = config
		self.arguments = arguments

		self.training_scene = training_scene
		self.training_objects = training_objects

		self.use_gae = arguments.get('use_gae')
		self.num_epochs = arguments.get('num_epochs')
		self.num_episodes = arguments.get('num_episodes')
		self.num_iters = arguments.get('num_iters')
		self.gamma = arguments.get('gamma')
		self.lamb = arguments.get('lamb')
		self.lr = arguments.get('lr')
		self.joint_loss = arguments.get('joint_loss')
		self.ec = arguments.get('ec')
		self.vc = arguments.get('vc')
		self.max_grad_norm = arguments.get('max_gradient_norm')
		self.dropout = arguments.get('dropout')
		self.decay = arguments.get('decay')
		self.reuse = arguments.get('share_latent')
		self.gpu_fraction = arguments.get('gpu_fraction')

		assert len(training_objects) == 2, "> 2 sharing agents are not supported yet."
		self.env = AI2ThorDumpEnv(training_scene, training_objects[0], config, arguments)
		
		sharing = self.env.h5_file["_".join(training_objects)][()].tolist()
		non_sharing = list(set(list(range(self.env.h5_file['locations'].shape[0]))) - set(sharing))

		self.sharing = dict(zip(sharing + non_sharing, [1] * len(sharing) + [0] * len(non_sharing)))

		self.rollouts = []
		for obj in training_objects:
			self.rollouts.append(Rollout(training_scene, obj, config, arguments))

		tf.reset_default_graph()

		self.PGNetworks = []
		for i in range(2):
			agent  = A2C(name='A2C_' + str(i), 
						state_size=self.env.features.shape[1], 
						action_size=self.env.action_space,
						history_size=arguments['history_size'],
						embedding_size=-1 if arguments['mode'] != 2 else 300,
						entropy_coeff=self.ec,
						value_function_coeff=self.vc,
						max_gradient_norm=self.max_grad_norm,
						dropout=self.dropout,
						joint_loss=self.joint_loss,
						learning_rate=self.lr,
						decay=self.decay,
						reuse=bool(self.reuse)
						)


			if self.decay:
				agent.set_lr_decay(self.lr, self.num_epochs * self.num_episodes * self.num_iters)

			
			print("\nInitialized network with {} trainable weights.".format(len(agent.find_trainable_variables('A2C_' + str(i), True))))
			self.PGNetworks.append(agent)


		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)

		self.sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))
		self.sess.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver()
		
		timer = "{}_{}_{}".format(str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-"), \
									training_scene, "_".join(training_objects))
		self.log_folder = os.path.join(arguments.get('logging'), timer)
		self.writer = tf.summary.FileWriter(self.log_folder)
		
		self.timer = timer

		test_name =  training_scene
		for i in range(len(training_objects)):
			tf.summary.scalar(test_name + "/" + training_objects[i] + "/rewards", self.PGNetworks[i].mean_reward)
			tf.summary.scalar(test_name + "/" + training_objects[i] + "/success_rate", self.PGNetworks[i].success_rate)
			tf.summary.scalar(test_name + "/" + training_objects[i] + "/redundants", self.PGNetworks[i].mean_redundant)

		self.write_op = tf.summary.merge_all()

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
		'''
		states = [
		    [---episode_1---],...,[---episode_n---]
		]
		same as actions, tasks, rewards, values, dones

		last_values = [
			episode_1, ...., episode_n]
		]
		same as redundants
		'''
		start = time.time()

		task_states, task_pis, task_actions, task_returns, task_advantages, tasks = [], [], [], [], [], []

		task_sc, task_rws, task_rdds = [], [], []

		for i in range(2):
			states, pis, actions, rewards, values, last_values, redundants = self.rollouts[i].rollout_batch(sess, self.PGNetworks[i], return_state_ids=True)

			success_count = 0			
			returns = []
			advantages = []

			for ep_idx, (ep_rewards, ep_states) in enumerate(zip(rewards, states)):
				assert len(ep_rewards) == len(ep_states)
				ep_dones = list(np.zeros_like(ep_rewards))

				if ep_rewards[-1] != self.config['success_reward']:
					last_value = last_values[ep_idx]
					assert last_value is not None
					ep_returns = self.discount_with_dones(ep_rewards + [last_value], ep_dones+[0], self.gamma)[:-1]
				else:
					success_count += 1
					last_value = 0
					ep_dones[-1] = 1
					ep_returns = self.discount_with_dones(ep_rewards, ep_dones, self.gamma)

				returns += ep_returns
				ep_values = values[ep_idx]

				if not self.use_gae:
					# Here we calculate advantage A(s,a) = R + yV(s') - V(s)
			    	# rewards = R + yV(s')
					advantages += list((np.array(ep_returns) - np.array(ep_values)).astype(np.float32))

				else:
					advantages += self.generalized_advantage_estimate(ep_rewards, ep_dones, ep_values, last_value, self.gamma, self.lamb)
			
			task_states += list(np.concatenate(states))
			task_pis += list(np.concatenate(pis))
			task_actions += list(np.concatenate(actions))
			task_returns += returns
			task_advantages += advantages
			tasks += [i] * len(returns)

			task_sc.append(success_count)
			task_rws.append(rewards)
			task_rdds.append(redundants)

		mean_policy = {}
		policies = {}
		for (s, a, pi, t) in zip(task_states, task_actions, task_pis, tasks):
			if self.sharing[s]:
				try:
					mean_policy[s, a].append(pi[a])
				except KeyError:
					mean_policy[s, a] = [pi[a]]

			try:
				policies[s, t].append(pi)
			except KeyError:
				policies[s, t] = [pi]


		
		for k in mean_policy.keys():
			mean_policy[k] = np.mean(mean_policy[k])

		for k in policies.keys():
			policies[k] = np.mean(policies[k], 0)

		batch_ss, batch_as, batch_ads, batch_rs = [], [], [], []
		share_ss, share_as, share_ads = [], [], []
		for task_index in range(2):
			batch_ss.append([])
			batch_as.append([])
			batch_ads.append([])
			batch_rs.append([])

			share_ss.append([])
			share_as.append([])
			share_ads.append([])

		for s, a, pi, r, ad, t in zip(task_states, task_actions, task_pis, task_returns, task_advantages, tasks):
			observation = self.env.state(s).reshape(1, -1).tolist()

			batch_ss[t].append(observation)
			batch_as[t].append(self.env.cv_action_onehot[a])
			batch_rs[t].append(r)

			if self.sharing[s]:
				batch_ads[t].append(ad * policies[s, t][a] / mean_policy[s, a])
				try:
					importance_weight = policies[s, 1 - t][a] / mean_policy[s, a]

					if importance_weight > 1.2:
						clipped_iw = 1.2
					elif importance_weight < 0.8:
						clipped_iw = 0.8
					else:
						clipped_iw = importance_weight

					if clipped_iw * ad < importance_weight * ad:
						share_ads[1 - t].append(clipped_iw * ad)
					else:
						share_ads[1 - t].append(importance_weight * ad)

						
					share_ss[1 - t].append(observation)
					share_as[1 - t].append(self.env.cv_action_onehot[a])				
				except KeyError:
					pass
			else:
				batch_ads[t].append(ad)


		return batch_ss,\
				 batch_as,\
				 batch_rs,\
				 batch_ads,\
				 share_ss,\
				 share_as,\
				 share_ads,\
				 task_rws,\
				 task_rdds,\
				 task_sc			 		
		
	def train(self):
		total_samples = [0, 0]
		errors = 0

		start = time.time()
		for epoch in range(self.num_epochs):

			batch_ss,\
			 batch_as,\
			 batch_rs,\
			 batch_ads,\
			 share_ss,\
			 share_as,\
			 share_ads,\
			 rewards,\
			 redundants,\
			 task_sc = self._make_batch(self.sess)
			#---------------------------------------------------------------------------------------------------------------------#	
			print('[{}-{}] Time elapsed: {:.3f}, epoch {}/{}, success_rate: {:.3f}'.format(\
				self.training_scene, self.training_objects, (time.time() - start)/3600, epoch + 1, \
				self.num_epochs, np.mean(task_sc) / self.num_episodes))

			sum_dict = {}
			assert len(batch_ss) == len(batch_as) == len(batch_rs) == len(batch_ads)
			assert len(share_ss) == len(share_as) == len(share_ads)
			
			for i in range(2):
				policy_loss, value_loss, _, _ = self.PGNetworks[i].learn(self.sess, actor_states=batch_ss[i] + share_ss[i],
																	advantages=batch_ads[i] + share_ads[i], 
																	actions=batch_as[i] + share_as[i],
																	critic_states=batch_ss[i], returns=batch_rs[i])
				
				sum_dict[self.PGNetworks[i].mean_reward] = np.sum(np.concatenate(rewards[i])) / self.num_episodes
				sum_dict[self.PGNetworks[i].success_rate] = task_sc[i] / self.num_episodes
				sum_dict[self.PGNetworks[i].mean_redundant] = np.mean(redundants[i])

				total_samples[i] += len(list(np.concatenate(rewards[i])))

				#---------------------------------------------------------------------------------------------------------------------#	
				

			# WRITE TF SUMMARIES
			#---------------------------------------------------------------------------------------------------------------------#	
			summary = self.sess.run(self.write_op, feed_dict = sum_dict)

			self.writer.add_summary(summary, np.mean(total_samples))
			self.writer.flush()
			#---------------------------------------------------------------------------------------------------------------------#	

		self.saver.save(self.sess, self.log_folder + "/my-model")
		self.sess.close()
		# SAVE MODEL
		#---------------------------------------------------------------------------------------------------------------------#	
		with open(self.log_folder + '/arguments.json', 'w') as outfile:
		    json.dump(self.arguments, outfile)

		print("\nElapsed time: {}".format((time.time() - start)/3600))	
		#---------------------------------------------------------------------------------------------------------------------#		