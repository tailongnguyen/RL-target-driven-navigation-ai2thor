import numpy as np           		
from utils import noise_and_argmax
from env.ai2thor_env import AI2ThorDumpEnv

class RolloutThread(object):
	
	def __init__(
		self,
		scene, 
		objects,
		task,
		policy, 
		config,
		custom_config):
	
		self.task = task
		self.noise_argmax = custom_config.get('noise_argmax') if custom_config.get('noise_argmax') is not None else config['noise_argmax']
		self.num_iters = custom_config.get('num_iters') if custom_config.get('num_iters') is not None else config['num_iters']

		self.policy = policy
		self.env = AI2ThorDumpEnv(scene, objects[task], config, custom_config)

	def rollout(self):
		states, tasks, actions, rewards_of_episode, next_states = [], [], [], [], []
		
		state, target = self.env.reset()
		step = 0

		while True:

			if self.noise_argmax:
				logit = self.policy[self.env.current_state_id, self.task, 0]
				action = noise_and_argmax(logit)
			else:
				pi = self.policy[self.env.current_state_id, self.task, 1]
				action = np.random.choice(range(len(pi)), p = np.array(pi)/ np.sum(pi))  # select action w.r.t the actions prob

			states.append(self.env.current_state_id)
			next_state, reward, done = self.env.step(action)
			
			# Store results
			tasks.append(self.task)
			actions.append(action)
			rewards_of_episode.append(reward)

			state = next_state
			next_states.append(self.env.current_state_id)

			step += 1

			if done or step > self.num_iters:   
				break

		return states, tasks, actions, rewards_of_episode, next_states