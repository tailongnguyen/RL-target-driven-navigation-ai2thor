import numpy as np           		
import sys

sys.path.append('..') # to access env package
from env.ai2thor_env import AI2ThorDumpEnv
from utils import noise_and_argmax

class RolloutThread(object):
	
	def __init__(
		self,
		sess,
		scene, 
		target,
		policy,
		embedding, 
		config,
		arguments):
		
		self.sess = sess
		self.noise_argmax = arguments.get('noise_argmax')
		self.num_iters = arguments.get('num_iters')

		self.policy = policy
		self.env = AI2ThorDumpEnv(scene, target, config, arguments)

		self.embedding = embedding
		if embedding is not None:
			self.task_input = embedding
		
	def rollout(self, return_state_ids=False):
		states, pis, actions, rewards, values, last_value = [], [], [], [], [], []
		
		state, score, target = self.env.reset()
		start = self.env.current_state_id
		step = 0

		while True:
			if self.embedding is not None:
				logit, p, v = self.sess.run(
							[self.policy.actor.logits, self.policy.actor.pi, self.policy.critic.value], 
							feed_dict={
								self.policy.actor.inputs: [state],
								self.policy.actor.task_input: [self.task_input],
								self.policy.critic.task_input: [self.task_input],
								self.policy.critic.inputs: [state]
							})
			else:	
				logit, p, v = self.sess.run(
								[self.policy.actor.logits, self.policy.actor.pi, self.policy.critic.value], 
								feed_dict={
									self.policy.actor.inputs: [state],
									self.policy.critic.inputs: [state]
								})

			if self.noise_argmax:
				action = noise_and_argmax(logit.ravel().tolist())
			else:
				pi = p.ravel().tolist()
				action = np.random.choice(range(len(pi)), p = np.array(pi)/ np.sum(pi))  # select action w.r.t the actions prob

			if return_state_ids:
				states.append(self.env.current_state_id)
			else:
				states.append(state)
				
			next_state, score, reward, done = self.env.step(action)
			
			# Store results
			pis.append(p.ravel().tolist())
			actions.append(action)
			rewards.append(reward)
			values.append(v)

			state = next_state

			step += 1

			if done or step > self.num_iters:   
				break

		if not done:
			if self.embedding is not None:
				last_value = self.sess.run(
							self.policy.critic.value, 
							feed_dict={
								self.policy.critic.inputs: [state],
								self.policy.critic.task_input: [self.task_input]
							})[0][0]
			else:
				last_value = self.sess.run(
								self.policy.critic.value, 
								feed_dict={
									self.policy.critic.inputs: [state]
								})[0][0]
		else:
			last_value = None

		end = self.env.current_state_id
		
		try:
			redundants = []
			for target_id in self.env.target_ids:
				redundants.append(step + self.env.shortest[end, target_id] - self.env.shortest[start, target_id])
		except AttributeError:
			redundants = [0]

		return states, pis, actions, rewards, values, last_value, min(redundants)