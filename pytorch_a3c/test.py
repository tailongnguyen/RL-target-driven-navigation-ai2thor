"""
Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/master/test.py

Contains the testing loop of the shared model within A3C (no optimisation/backprop needed)
Usually this is run concurrently while training occurs and is useful for tracking progress. But to
save resources we can choose to only test every args.test_sleep_time seconds.
"""

import time
from collections import deque

import pickle
import torch
import torch.nn.functional as F
import numpy as np
import sys
import cv2

sys.path.append('..') # to access env package

from env.ai2thor_env import AI2ThorDumpEnv
from model import ActorCritic

def test(testing_scene, test_object, rank, shared_model, results, config, arguments=dict()):
    torch.manual_seed(arguments['seed'] + rank)

    env = AI2ThorDumpEnv(testing_scene, test_object, config, arguments)

    model = shared_model
    if model is not None:
        if arguments['use_gpu']:
            model.cuda()

        model.eval()

    state, score, target = env.reset()
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    results[rank] = 0

    for ep in range(arguments['num_epochs']):
        for step in range(arguments['num_iters']):
            if model is not None:
                with torch.no_grad():
                    value, logit = model(state, score, target)
                prob = F.softmax(logit, dim=-1)
                # action = prob.max(1, keepdim=True)[1].numpy()
                action = prob.multinomial(num_samples=1).detach().numpy()[0, 0]

            else:
                action = np.random.choice(range(arguments['action_size']))

            state, score, reward, done = env.step(action)
            # a quick hack to prevent the agent from stucking
            # i.e. in test mode an agent can repeat an action ad infinitum
            actions.append(action)
            if actions.count(actions[0]) == actions.maxlen:
                # print('In test. Episode over because agent repeated action {} times'.format(
                                                                                    # actions.maxlen))
                break

            if done:                
                actions.clear()
                results[rank] += 1
                state, score, target = env.reset()
                break

        # print('[P-{}] ep {}/{}: {}'.format(rank, ep+1, arguments['num_epochs'], 'fail' if not done else 'success'))

    results[rank] = results[rank] / arguments['num_epochs']

def live_test(testing_scene, test_object, shared_model, config, arguments=dict()):
    env = AI2ThorDumpEnv(testing_scene, test_object, config, arguments)

    model = shared_model
    if model is not None:
        if arguments['use_gpu']:
            model.cuda()

        model.eval()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)

    while 1:
        state, score, target = env.reset()
        start = env.current_state_id
        done = True
        actions.clear()

        for step in range(arguments['num_iters']):
            ob = env.observations[env.current_state_id]
            
            cv2.imshow("Live Test", ob[:,:,::-1])
            time.sleep(0.3)
            k = cv2.waitKey(33) 
            if k == ord('r'): # press q to escape
                break
            elif k == ord('q'): # press q to escape
                sys.exit("End live test.")


            if model is not None:
                with torch.no_grad():
                    value, logit = model(state, score, target)
                prob = F.softmax(logit, dim=-1)
                # action = prob.max(1, keepdim=True)[1].numpy()[0, 0]
                action = prob.multinomial(num_samples=1).detach().numpy()[0, 0]

            else:
                action = np.random.choice(range(arguments['action_size']))

            print("Action: {}".format(['Move Forward', 'Move Backward', 'Turn Right', 'Turn Left'][action]))
            state, score, reward, done = env.step(action)
            if env.collided:
                print("Collision occurs.")
            # a quick hack to prevent the agent from stucking
            # i.e. in test mode an agent can repeat an action ad infinitum
            actions.append(action)
            if actions.count(actions[0]) == actions.maxlen:
                # print('In test. Episode over because agent repeated action {} times'.format(
                                                                                    # actions.maxlen))
                break

            if done:                
                break

        if not done:
            print("Fail")
        else:
            print("Success with {} redundant steps.".format(step + 1 - env.shortest[start, env.current_state_id]))            