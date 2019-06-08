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

from env.ai2thor_env import AI2ThorDumpEnv, MultiSceneEnv
from model import ActorCritic

def test(testing_scene, test_object, rank, shared_model, results, config, arguments=dict()):
    torch.manual_seed(arguments['seed'] + rank)

    env = AI2ThorDumpEnv(testing_scene, test_object, config, arguments, arguments['seed'] + rank)
    print("Finding {} in {}, {}".format(test_object, testing_scene, env.target_locs))

    if shared_model is not None:
        gpu_id = arguments['gpu_ids'][rank % len(arguments['gpu_ids'])]
        # gpu_id = -1

        model = ActorCritic(config, arguments, gpu_id)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                model = model.cuda()
                model.load_state_dict(shared_model.state_dict())

                print("[P{}] loaded model into cuda {}".format(rank, gpu_id))
        else:
            model.load_state_dict(shared_model.state_dict())
            print("[P{}] loaded model".format(rank))

        model.eval()

    state, score, target = env.reset()
    done = True
    
    starting = env.current_state_id
    results[rank] = 0
    
    for ep in range(1000):
        agent_step = 0
        for step in range(arguments['num_iters']):
            if model is not None:
                with torch.no_grad():
                    value, logit = model(state, score, target)
                prob = F.softmax(logit, dim=-1)
                action = prob.max(1, keepdim=True)[1].cpu().numpy()
                # action = prob.multinomial(num_samples=1).detach().cpu().numpy()[0, 0]

            else:
                action = np.random.choice(range(arguments['action_size']))

            state, score, reward, done = env.step(action)
            ending = env.current_state_id

            if action < 2:
                agent_step += 1
            
            if done:                
                results[rank] += env.shortest[ending, starting] / max(agent_step, env.shortest[ending, starting])
                state, score, target = env.reset()
                break

    results[rank] = results[rank] / 1000

def test_multi(testing_scene, rank, shared_model, results, config, arguments=dict()):
    torch.manual_seed(arguments['seed'] + rank)

    env = MultiSceneEnv(testing_scene, config, arguments, arguments['seed'] + rank)

    # gpu_id = arguments['gpu_ids'][rank % len(arguments['gpu_ids'])]
    gpu_id = -1
    print("Done initalizing process {}: {}! Use gpu: {}".format(rank, testing_scene, 'yes' if gpu_id >= 0 else 'no'))

    if shared_model is not None:
        # gpu_id = -1

        model = ActorCritic(config, arguments, gpu_id)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                model = model.cuda()
                model.load_state_dict(shared_model.state_dict())

                # print("[P{}] loaded model into cuda {}".format(rank, gpu_id))
        else:
            model.load_state_dict(shared_model.state_dict())
            # print("[P{}] loaded model".format(rank))

        model.eval()
        
    else:
        model = None
    state, score, target = env.reset()
    done = True

    for ep in range(1000):
        state, score, target = env.reset()
        agent_step = 0
        starting = env.current_state_id

        for step in range(arguments['num_iters']):
            if model is not None:
                with torch.no_grad():
                    value, logit = model(state, score, target)
                prob = F.softmax(logit, dim=-1)
                action = prob.max(1, keepdim=True)[1].cpu().numpy()
                # action = prob.multinomial(num_samples=1).detach().cpu().numpy()[0, 0]

            else:
                action = np.random.choice(range(arguments['action_size']))

            state, score, reward, done = env.step(action)
            ending = env.current_state_id
            
            if action < 2:
                agent_step += 1

            if done:                
                break

        if not done:
            tm = results[target]
            tm.append(0)
            results[target] = tm
        else:
            if max(agent_step, env.shortest[ending, starting]) > 0:
                tm = results[target]
                tm.append(env.shortest[ending, starting] / max(agent_step, env.shortest[ending, starting]))
                results[target] = tm

def live_test(testing_scene, test_objects, shared_model, config, arguments=dict()):

    model = shared_model
    if model is not None:
        model.eval()

    test_object = np.random.choice(test_objects)
    env = AI2ThorDumpEnv(testing_scene, test_object, config, arguments)
    print(arguments['angle'])

    new_test_object = None
    while 1:
        if new_test_object is not None and new_test_object != test_object:
            print("Finding {} ..".format(new_test_object))
            env = AI2ThorDumpEnv(testing_scene, new_test_object, config, arguments)
        else:
            print("Finding {} ..".format(test_object))

        state, score, target = env.reset()
        start = env.current_state_id
        done = True
        stop = 0

        for step in range(arguments['num_iters']):
            ob = env.observations[env.current_state_id]
            
            cv2.imshow("Live Test", cv2.resize(ob[:,:,::-1], (400, 400)))
            time.sleep(0.1)
            k = cv2.waitKey(33) 

            if k == ord('r'): # press q to escape
                new_test_object_id = int(input("Specify target: {}\n".format(list(zip(range(len(test_objects)), test_objects)))))
                new_test_object = test_objects[new_test_object_id]
                break
            elif k == ord('q'): # press q to escape
                sys.exit("End live test.")


            if model is not None:
                with torch.no_grad():
                    value, logit = model(state, score, target)
                prob = F.softmax(logit, dim=-1)
                action = prob.max(1, keepdim=True)[1].numpy()[0, 0]
                # action = prob.multinomial(num_samples=1).detach().numpy()[0, 0]

            else:
                action = np.random.choice(range(arguments['action_size']))

            print("Action: {}".format(['Move Forward', 'Move Backward', 'Turn Right', 'Turn Left'][action]))
            state, score, reward, done = env.step(action)
            if env.collided:
                print("Collision occurs.")
            # a quick hack to prevent the agent from stucking
            # i.e. in test mode an agent can repeat an action ad infinitum
        
            if done: 
                stop += 1
                if stop == 2:               
                    new_test_object_id = int(input("Specify target: {}\n".format(list(zip(range(len(test_objects)), test_objects)))))
                    new_test_object = test_objects[new_test_object_id]
                    stop = 0
                    break

        if not done:
            print("Fail")
        else:
            print("Success with {} redundant steps.".format(step + 1 - env.shortest[start, env.current_state_id]))            
