import numpy as np           		
from utils import noise_and_argmax
from env.ai2thor_env import AI2ThorDumpEnv

class RolloutThread(object):
	
	def __init__(
		self,
		sess,
		scene, 
		target,
		policy, 
		config,
		arguments):
		
		self.sess = sess
		self.noise_argmax = arguments.get('noise_argmax')
		self.num_iters = arguments.get('num_iters')

		self.policy = policy
		self.env = AI2ThorDumpEnv(scene, target, config, arguments)

	def rollout(self):
		states, logits, actions, rewards, values, last_value = [], [], [], [], [], []
		
		state, target = self.env.reset()
		step = 0

		while True:
			logit, p, v = self.sess.run(
							[self.policy.actor.logits, self.policy.actor.pi, self.policy.critic.value], 
							feed_dict={
								self.policy.actor.inputs: [state],
								self.policy.critic.inputs: [state]
							})

			if self.noise_argmax:
				action = noise_and_argmax(logit)
			else:
				action = np.random.choice(range(len(pi)), p = np.array(pi)/ np.sum(pi))  # select action w.r.t the actions prob

			states.append(state)
			next_state, reward, done = self.env.step(action)
			
			# Store results
			logits.append(logit[0])
			actions.append(action)
			rewards.append(reward)
			values.append(v)

			state = next_state

			step += 1

			if done or step > self.num_iters:   
				break

		if not done:
			last_value = self.sess.run(
							self.policy.critic.value, 
							feed_dict={
								self.policy.critic.inputs: [state]
							})[0]
		else:
			last_value = None

		return states, logits, actions, rewards, values, last_value