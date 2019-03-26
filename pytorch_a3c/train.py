"""
Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

Contains the train code run by each A3C process on either Atari or AI2ThorEnv.
For initialisation, we set up the environment, seeds, shared model and optimizer.
In the main training loop, we always ensure the weights of the current model are equal to the
shared model. Then the algorithm interacts with the environment arguments.num_steps at a time,
i.e it sends an action to the env for each state and stores predicted values, rewards, log probs
and entropies to be used for loss calculation and backpropagation.
After arguments.num_steps has passed, we calculate advantages, value losses and policy losses using
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
import os
import sys
import pickle
import sys

sys.path.append('..') # to access env package

from env.ai2thor_env import AI2ThorDumpEnv, MultiSceneEnv
from model import ActorCritic


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()

def train(training_scene, train_object, single_target, rank, shared_model, scheduler, counter, lock, config, arguments=dict(), optimizer=None):
    torch.manual_seed(arguments['seed'] + rank)
    # To prevent out of memory
    if (arguments['lstm'] and rank < 8) or (arguments['use_gcn'] and rank < 10):
        arguments.update({"gpu_ids": [-1]})

    gpu_id = arguments['gpu_ids'][rank % len(arguments['gpu_ids'])]

    if gpu_id >= 0:
        torch.cuda.manual_seed(arguments['seed'] + rank)

    if optimizer is None:
        optimizer = optim.RMSprop(shared_model.parameters(), lr=arguments['lr'],  alpha=0.99, eps=0.1)
        # optimizer = optim.Adam(shared_model.parameters(), lr=arguments['lr'])

    if arguments['easy']:
        env = AI2ThorDumpEnv(training_scene, train_object, single_target[train_object], config, arguments)
    else:
        env = AI2ThorDumpEnv(training_scene, train_object, None, config, arguments)
    
    state, score, target = env.reset()
    starting = env.current_state_id
    done = True
    print("Done initalizing process {}. Now find {} in {}! Use gpu: {}".format(rank, env.target, env.scene, 'yes' if gpu_id >= 0 else 'no'))

    model = ActorCritic(config, arguments, gpu_id)
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            model = model.cuda()

    model.train()

    # monitoring
    total_reward_for_num_steps_list = []
    redundancies = []
    success = []
    avg_entropies = []

    start = time.time()

    episode_length = 0
    for epoch in range(arguments['num_epochs']):
        # Sync with the shared model
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                model.load_state_dict(shared_model.state_dict())
        else:
            model.load_state_dict(shared_model.state_dict())

        if scheduler is not None:
            scheduler.step()
        
        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(arguments['num_iters']):
            episode_length += 1

            value, logit = model(state, score, target)
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            action_int = action.cpu().numpy()[0][0].item()
            state, score, reward, done = env.step(action_int)

            if done:
                success.append(1)
            elif episode_length >= arguments['max_episode_length']:
                success.append(0)

            done = done or episode_length >= arguments['max_episode_length']

            with lock:
                counter.value += 1

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            ending = env.current_state_id
            if done:
                state, score, target = env.reset()
                if arguments['lstm']:
                    model.reset_lstm()
                    
                print('[P-{}] Episode length: {}. Total reward: {:.3f}. Time elapsed: {:.3f}'\
                        .format(rank, episode_length, sum(rewards), (time.time() - start) / 3600))

                episode_length = 0
                break

        if not done:
            success.append(0)

        # No interaction with environment below.
        # Monitoring
        total_reward_for_num_steps_list.append(sum(rewards))
        redundancies.append(step + 1 - env.shortest[ending, starting])
        avg_entropies.append(torch.tensor(entropies).numpy().mean())

        # Backprop and optimisation
        R = torch.zeros(1, 1)
        if not done:  # to change last reward to predicted value to ....
            value, _, = model(state, score, target)
            R = value.detach()
        
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        values.append(R)

        policy_loss = 0
        value_loss = 0

        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()

        for i in reversed(range(len(rewards))):
            if arguments['norm_reward']:
                R = arguments['gamma'] * R + rewards[i] * arguments['num_iters'] / (step + 1)
            else:
                R = arguments['gamma'] * R + rewards[i]

            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            if arguments['use_gae']:
                # Generalized Advantage Estimation
                delta_t = rewards[i] + arguments['gamma'] * values[i + 1] - values[i]
                gae = gae * arguments['gamma'] * arguments['tau'] + delta_t

                policy_loss = policy_loss - log_probs[i] * gae.detach() - \
                              arguments['ec'] * entropies[i]
            else:
                policy_loss = policy_loss - log_probs[i] * advantage.detach() - \
                              arguments['ec'] * entropies[i]

        optimizer.zero_grad()

        (policy_loss + arguments['vc'] * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), arguments['max_grad_norm'])

        ensure_shared_grads(model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            with open('training-history/{}/{}_{}_{}.pkl'.format(arguments['about'], training_scene, train_object, rank), 'wb') as f:
                pickle.dump({"rewards": total_reward_for_num_steps_list, 
                            "success_rate": success, 'redundancies': redundancies,
                            "entropies": avg_entropies}, f, pickle.HIGHEST_PROTOCOL)

            torch.save(model.state_dict(), "training-history/{}/net_{}.pth".format(arguments['about'], train_object))

    with lock:
        print("Done in steps {}th".format(counter.value))

def train_multi(training_scene, rank, shared_model, scheduler, counter, lock, config, arguments=dict(), optimizer=None):
    torch.manual_seed(arguments['seed'] + rank)

    env = MultiSceneEnv(training_scene, config, arguments)
    
    model = ActorCritic(config, arguments)

    if optimizer is None:
        optimizer = optim.RMSprop(shared_model.parameters(), lr=arguments['lr'], alpha=0.99, eps=0.1)
        # optimizer = optim.Adam(shared_model.parameters(), lr=arguments['lr'])

    model.train()

    state, score, new_target = env.reset()
    starting = env.current_state_id
    done = True
    print("Done resetting. Now find {} in {}!".format(env.target, env.scene))

    # monitoring
    total_reward_for_num_steps_list = {}
    redundancies = []
    avg_entropies = []
    success = []

    start = time.time()

    episode_length = 0
    for epoch in range(arguments['num_epochs']):
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if scheduler is not None:
            scheduler.step()
        
        values = []
        log_probs = []
        rewards = []
        entropies = []

        target = new_target
        for step in range(arguments['num_iters']):
            episode_length += 1

            value, logit = model(state, score, target)
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            action_int = action.cpu().numpy()[0][0].item()
            state, score, reward, done = env.step(action_int)

            if done:
                success.append(1)
            elif episode_length >= arguments['max_episode_length']:
                success.append(0)

            done = done or episode_length >= arguments['max_episode_length']

            with lock:
                counter.value += 1

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            ending = env.current_state_id

            if done:
                state, score, new_target = env.reset()
                print('[P-{}] Episode length: {}. Total reward: {:.3f}. Time elapsed: {:.3f}'\
                        .format(rank, episode_length, sum(rewards), (time.time() - start) / 3600))

                episode_length = 0
                break

        if not done:
            success.append(0)

        # No interaction with environment below.
        # Monitoring
        try:
            total_reward_for_num_steps_list[target].append(sum(rewards))
        except KeyError:
            total_reward_for_num_steps_list[target] = [sum(rewards)]

        redundancies.append(step + 1 - env.shortest[ending, starting])
        avg_entropies.append(torch.tensor(entropies).numpy().mean())

        # Backprop and optimisation
        R = torch.zeros(1, 1)
        if not done:  # to change last reward to predicted value to ....
            value, _, = model(state, score, target)
            R = value.detach()
    
        values.append(R)

        policy_loss = 0
        value_loss = 0
        # import pdb;pdb.set_trace() # good place to breakpoint to see training cycle
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            if arguments['norm_reward']:
                R = arguments['gamma'] * R + rewards[i] * arguments['num_iters'] / (step + 1)
            else:
                R = arguments['gamma'] * R + rewards[i]

            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            if arguments['use_gae']:
                # Generalized Advantage Estimation
                delta_t = rewards[i] + arguments['gamma'] * values[i + 1] - values[i]
                gae = gae * arguments['gamma'] * arguments['tau'] + delta_t

                policy_loss = policy_loss - log_probs[i] * gae.detach() - \
                              arguments['ec'] * entropies[i]
            else:
                policy_loss = policy_loss - log_probs[i] * advantage.detach() - \
                              arguments['ec'] * entropies[i]

        optimizer.zero_grad()

        (policy_loss + arguments['vc'] * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), arguments['max_grad_norm'])

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            with open('training-history/{}/{}_{}.pkl'.format(arguments['about'], training_scene, rank), 'wb') as f:
                pickle.dump({"rewards": total_reward_for_num_steps_list, 
                            "success_rate": success, 'redundancies': redundancies,
                            "entropies": avg_entropies}, f, pickle.HIGHEST_PROTOCOL)

            torch.save(model.state_dict(), "training-history/{}/net_{}.pth".format(arguments['about'], training_scene))

    with lock:
        print("Done in steps {}th".format(counter.value))
    