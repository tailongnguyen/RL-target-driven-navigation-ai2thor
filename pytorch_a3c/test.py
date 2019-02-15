"""
Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/master/test.py

Contains the testing loop of the shared model within A3C (no optimisation/backprop needed)
Usually this is run concurrently while training occurs and is useful for tracking progress. But to
save resources we can choose to only test every args.test_sleep_time seconds.
"""

import time
from collections import deque

import torch
import torch.nn.functional as F

from env.ai2thor_env import AI2ThorDumpEnv
from model import ActorCritic


def test(testing_scenes, test_objects, rank, args, shared_model, config, counter, use_gpu=False):
    torch.manual_seed(args.seed + rank)

    
    env = AI2ThorDumpEnv(config, testing_scenes, test_objects)
    env.seed(args.seed + rank)

    model = ActorCritic(config, env.action_space, use_gpu=use_gpu)
    if use_gpu:
        model.cuda()
    model.eval()

    state, target = env.reset()
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
        
        with torch.no_grad():
            value, logit = model(state, target)
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done = env.step(action[0, 0])
        done = done or episode_length >= args.num_steps
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        # i.e. in test mode an agent can repeat an action ad infinitum
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            print('In test. Episode over because agent repeated action {} times'.format(
                                                                                actions.maxlen))
            done = True

        if done:
            print("[TEST] Time {}, num steps over all threads {}, FPS {:.0f}, episode reward {:.3f}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state, target = env.reset()
            time.sleep(args.test_sleep_time)
