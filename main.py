"""
Adapted from: https://github.com/ikostrikov/pytorch-a3c/blob/master/main.py
The main file needed within a3c. Runs of the train and test functions from their respective files.
Example of use:
`cd algorithms/a3c`
`python main.py`

Runs A3C on our AI2ThorEnv wrapper with default params (4 processes). Optionally it can be
run on any atari environment as well using the --atari and --atari-env-name params.
"""

from __future__ import print_function

import argparse
import os
import numpy as np
import torch
import torch.multiprocessing as mp
import json

from env.ai2thor_env import AI2ThorEnv, AI2ThorDumpEnv
from optimizers import SharedAdam
from model import ActorCritic
from test import test
from train import train

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--about', type=str, default="training A3C",
                    help='description about training')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.96,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--room_id', type=int, default=0,
                    help='room id (default: 0)')
parser.add_argument('--test-sleep-time', type=int, default=200,
                    help='number of seconds to wait before testing again (default: 200)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-steps', type=int, default=50,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--num-epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('-sync', '--synchronous', dest='synchronous', action='store_true',
                    help='Useful for debugging purposes e.g. import pdb; pdb.set_trace(). '
                         'Overwrites args.num_processes as everything is in main thread. '
                         '1 train() function is run and no test()')
parser.add_argument('-async', '--asynchronous', dest='synchronous', action='store_false')
parser.add_argument('--config_file', type=str, default="config.json")
parser.set_defaults(synchronous=False)

ALL_ROOMS = {
    0: "Kitchens",
    1: "Living Rooms",
    2: "Bedrooms",
    3: "Bathrooms"
}

def read_config(config_path):
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    return config

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    if os.environ['CUDA_VISIBLE_DEVICES'] != "":
        use_gpu = True
    else:
        use_gpu = False

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    config = read_config(args.config_file)
    room = config['rooms'][ALL_ROOMS[args.room_id]]
    all_scenes = room['scenes']
    train_objects = room['train_objects']
    test_objects = room['test_objects']

    # np.random.shuffle(all_scenes)
    # training_scenes = all_scenes[:20]
    # validation_scenes = all_scenes[20:25]
    # testing_scenes = all_scenes[25:]

    training_scenes = all_scenes[:1]

    # env = AI2ThorEnv(config, training_scenes, train_objects)
    
    shared_model = ActorCritic(config, 6, train_resnet=False, use_gpu=use_gpu)
    shared_model.share_memory()

    if use_gpu:
        shared_model.cuda()

    # env.close()  # above env initialisation was only to find certain params needed

    if args.no_shared:
        optimizer = None
    else:
        optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    if not args.synchronous:
        # test runs continuously and if episode ends, sleeps for args.test_sleep_time seconds
        # p = mp.Process(target=test, args=(training_scenes, train_objects, args.num_processes, args, shared_model, config, counter, use_gpu))
        # p.start()
        # processes.append(p)

        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(training_scenes, train_objects, rank, args, shared_model, counter, lock, config, optimizer, use_gpu))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        torch.save(shared_model.state_dict(), "training-history/")
    else:
        print("Start training synchronously")
        rank = 0
        # test(args.num_processes, args, shared_model, counter)  # for checking test functionality
        train(training_scenes, train_objects, rank, args, shared_model, counter, lock, config, optimizer, use_gpu)  # run train on main thread
