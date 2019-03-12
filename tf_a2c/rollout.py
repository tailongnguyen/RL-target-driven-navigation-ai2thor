import threading
from rollout_thread import RolloutThread

class Rollout(object):
	
	def __init__(
		self,
		training_scene, 
		training_object,
		config,
		arguments,
		embedding=None):
		
		self.config = config
		self.arguments = arguments
		self.embedding = embedding
		
		self.num_episodes = arguments.get('num_episodes')

		self.training_scene = training_scene
		self.training_object = training_object

		self.states, self.pis, self.actions, self.rewards, self.values, self.last_values, self.redundants = \
										[self.holder_factory(self.num_episodes) for i in range(7)]

	def _rollout_process(self, index, sess, policy, return_state_ids):
		thread_rollout = RolloutThread(sess=sess, scene=self.training_scene, target=self.training_object,
										policy=policy, embedding=self.embedding, 
										config=self.config, arguments=self.arguments)

		ep_states, ep_logits, ep_actions, ep_rewards, ep_values, ep_last_value, ep_redundant = thread_rollout.rollout(return_state_ids)
		
		self.states[index] = ep_states
		self.pis[index] = ep_logits
		self.actions[index] = ep_actions
		self.rewards[index] = ep_rewards
		self.values[index] = ep_values
		self.last_values[index] = ep_last_value
		self.redundants[index] = ep_redundant

	def holder_factory(self, num_episodes):
		return [[] for j in range(num_episodes)] 

	def rollout_batch(self, sess, policy, return_state_ids=False):
		self.states, self.pis, self.actions, self.rewards, self.values, self.last_values, self.redundants = \
										[self.holder_factory(self.num_episodes) for i in range(7)]

		train_threads = []
		
		for i in range(self.num_episodes):
			train_threads.append(threading.Thread(target=self._rollout_process, args=(i, sess, policy, return_state_ids)))

		# start each training thread
		for t in train_threads:
			t.start()

		# wait for all threads to finish
		for t in train_threads:
			t.join()		

		return self.states, self.pis, self.actions, self.rewards, self.values, self.last_values, self.redundants