"""
Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

Contains the train code run by each A3C process on either Atari or AI2ThorEnv.
For initialisation, we set up the environment, seeds, shared model and optimizer.
In the main training loop, we always ensure the weights of the current model are equal to the
shared model. Then the algorithm interacts with the environment args.num_steps at a time,
i.e it sends an action to the env for each state and stores predicted values, rewards, log probs
and entropies to be used for loss calculation and backpropagation.
After args.num_steps has passed, we calculate advantages, value losses and policy losses using
Generalized Advantage Estimation (GAE) with the entropy loss added onto policy loss to encourage
exploration. Once these losses have been calculated, we add them all together, backprop to find all
gradients and then optimise with Adam and we go back to the start of the main training loop.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import json
import sys
import pickle

from env.ai2thor_env import AI2ThorDumpEnv
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(training_scenes, train_objects, rank, args, shared_model, counter, lock, config, custom_config=dict(), optimizer=None, use_gpu=False):
    torch.manual_seed(args.seed + rank)

    env = AI2ThorDumpEnv(training_scenes, train_objects, config, custom_config)
    env.seed(args.seed + rank)

    model = ActorCritic(env.action_space, config, custom_config, train_resnet=custom_config.get('train_resnet') or config['train_resnet'], use_gpu=use_gpu)
    if use_gpu:
        model.cuda()

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state, target = env.reset()
    done = True
    print("Done resetting. Now find {} in {}!".format(env.target, env.scene_id))

    # monitoring
    total_reward_for_num_steps_list = []
    avg_reward_for_num_steps_list = []
    path_length = {}

    total_length = 0
    start = time.time()

    for epoch in range(args.num_epochs):
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        
        values = []
        log_probs = []
        rewards = []
        entropies = []
        episode_length = 0

        while True:
            episode_length += 1
            total_length += 1

            value, logit = model(state, target)
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            action_int = action.cpu().numpy()[0][0].item()
            state, reward, done = env.step(action_int)

            if done:
                if env.current_state_id not in path_length:
                    path_length[env.current_state_id] = [episode_length]
                else:
                    path_length[env.current_state_id].append(episode_length)
            elif episode_length == args.num_steps:
                if env.current_state_id not in path_length:
                    path_length[env.current_state_id] = [-1]
                else:
                    path_length[env.current_state_id].append(-1)

            done = done or episode_length >= args.num_steps

            with lock:
                counter.value += 1

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                total_reward_for_episode = sum(rewards)
                state, target = env.reset()
                print('[P-{}] Episode length: {}. Total length: {}. Total reward: {:.3f}. Time elapsed: {:.3f}'\
                        .format(rank, episode_length, total_length, total_reward_for_episode, (time.time() - start) / 3600))

                break


        # No interaction with environment below.
        # Monitoring
        total_reward_for_num_steps_list.append(sum(rewards))
        avg_reward_for_num_steps = sum(rewards) / len(rewards)
        avg_reward_for_num_steps_list.append(avg_reward_for_num_steps)

        # Backprop and optimisation
        R = torch.zeros(1, 1)
        if not done:  # to change last reward to predicted value to ....
            value, _, = model(state, target)
            R = value.detach()

        if use_gpu:
            values.append(R.type(torch.cuda.FloatTensor))
        else:
            values.append(R)

        policy_loss = 0
        value_loss = 0
        # import pdb;pdb.set_trace() # good place to breakpoint to see training cycle
        gae = torch.zeros(1, 1)

        if use_gpu:
            R = R.type(torch.cuda.FloatTensor)
            gae = gae.type(torch.cuda.FloatTensor)

        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - \
                          args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

    with open('training-history/{}.pkl'.format(args.about), 'wb') as f:
        pickle.dump(path_length, f, pickle.HIGHEST_PROTOCOL)
    