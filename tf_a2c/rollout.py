import threading
from rollout_thread import RolloutThread

class Rollout(object):
	
	def __init__(
		self,
		training_scene, 
		training_objects,
		config,
		custom_config):
		
		self.config = config
		self.custom_config = custom_config
		
		self.num_task = custom_config.get('num_task') if custom_config.get('num_task') is not None else config['num_task']
		self.num_episodes = custom_config.get('num_episodes') if custom_config.get('num_episodes') is not None else config['num_episodes']

		self.training_scene = training_scene
		self.training_objects = training_objects

		self.states, self.tasks, self.actions, self.rewards, self.next_states = \
										[self.holder_factory(self.num_task, self.num_episodes) for i in range(5)]

	def _rollout_process(self, task, index, current_policy):
		thread_rollout = RolloutThread(scene=self.training_scene, objects=self.training_objects, task=task,
									policy=current_policy, config=self.config, custom_config=self.custom_config)

		ep_states, ep_tasks, ep_actions, ep_rewards, ep_next_states = thread_rollout.rollout()
		
		self.states[task][index] = ep_states
		self.tasks[task][index] = ep_tasks
		self.actions[task][index] = ep_actions
		self.rewards[task][index] = ep_rewards
		self.next_states[task][index] = ep_next_states

	def holder_factory(self, num_task, num_episodes):
		return [ [ [] for j in range(num_episodes)] for i in range(num_task) ]

	def rollout_batch(self, current_policy):
		self.states, self.tasks, self.actions, self.rewards, self.next_states = \
										[self.holder_factory(self.num_task, self.num_episodes) for i in range(5)]

		train_threads = []
		
		for task in range(self.num_task):
			for i in range(self.num_episodes):
				train_threads.append(threading.Thread(target=self._rollout_process, args=(task, i, current_policy, )))

		# start each training thread
		for t in train_threads:
			t.start()

		# wait for all threads to finish
		for t in train_threads:
			t.join()		

		return self.states, self.tasks, self.actions, self.rewards, self.next_states