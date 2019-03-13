"""
Adapted from: https://github.com/ikostrikov/pytorch-a3c/blob/master/main.py
The main file needed within a3c. Runs of the train and test functions from their respective files.
Example of use:
`cd algorithms/a3c`
`python main.py`

Runs A3C on our AI2ThorEnv wrapper with default params (4 processes). Optionally it can be
run on any atari environment as well using the --atari and --atari-env-name params.
"""

import argparse
import os
import numpy as np
import torch
import torch.multiprocessing as mp
import json

from env.ai2thor_env import AI2ThorDumpEnv
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
parser.add_argument('--lr', type=float, default=7e-4,
                    help='learning rate (default: 7e-4)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.96,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--ec', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--vc', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--lr_decay', type=int, default=0,
                    help='whether to use learning rate decay')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--room_id', type=int, default=0,
                    help='room id (default: 0)')
parser.add_argument('--test', type=int, default=0,
                    help='whether to activate testing phase')
parser.add_argument('--action_size', type=int, default=4,
                    help='number of possible actions')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-iters', type=int, default=100,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--num-epochs', type=int, default=10000,
                    help='number of epochs to train in each thread')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--train-resnet', type=int, default=0,
                    help='whether to include resnet into training')
parser.add_argument('--use-gpu', type=int, default=0,
                    help='whether to use gpu to train')
parser.add_argument('--history-size', type=int, default=4,
                    help='whether to stack frames')
parser.add_argument('--embed', type=int, default=0,
                    help='embedding mode: 0 for onehot, 1 for fasttext')
parser.add_argument('--use_gcn', type=int, default=0,
                    help='whether to include gcn')
parser.add_argument('--anti_col', type=int, default=0,
                    help='whether to include collision penalty to rewarding scheme')
parser.add_argument('--no-shared', type=int, default=0,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--scene_id', type=int, default=0,
                        help='scene id (default: 0)')

parser.add_argument('--config_file', type=str, default="config.json")
parser.add_argument('--weights', type=str)
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    config = read_config(args.config_file)
    room = config['rooms'][ALL_ROOMS[args.room_id]]

    training_scene = "FloorPlan{}".format(args.scene_id)
    trainable_objects = config["picked"][training_scene]['train']

    print(list(zip(range(len(trainable_objects)), trainable_objects)))

    target_ids = [int(i.strip()) for i in input("Please specify target id: ").split(",")]
    training_objects = [trainable_objects[target_id] for target_id in target_ids]
    num_thread_each = args.num_processes // len(training_objects)
    object_threads = []

    for obj in training_objects:
        object_threads += [obj] * num_thread_each

    object_threads += [np.random.choice(training_objects)] * (args.num_processes - len(object_threads))

    if not os.path.isdir("training-history/{}".format(args.about)):
        os.mkdir("training-history/{}".format(args.about))

    print("Start training agent to find {} in {}".format(training_objects, training_scene))

    shared_model = ActorCritic(config, vars(args))
    shared_model.share_memory()

    if args.use_gpu:
        shared_model.cuda()

    # env.close()  # above env initialisation was only to find certain params needed

    scheduler = None
    if args.no_shared:
        optimizer = None
    else:
        optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()
        if args.lr_decay:
            decay_step = (args.lr - 1e-6) / (args.num_epochs * args.num_processes)
            lambda1 = lambda epoch: args.lr - epoch * decay_step
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    processes = []


    if not args.test:
        counter = mp.Value('i', 0)
        lock = mp.Lock()
        # test runs continuously and if episode ends, sleeps for args.test_sleep_time seconds
        # p = mp.Process(target=test, args=(training_scene, training_object, args.num_processes, \
        #                 shared_model, config, counter, vars(args)))
        # p.start()
        # processes.append(p)

        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(training_scene, object_threads[rank], rank, shared_model, \
                            scheduler, counter, lock, config, vars(args), optimizer))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        with open('training-history/{}/arguments.json'.format(args.about), 'w') as outfile:
            json.dump(vars(args), outfile)
    else:
        print("Start testing ..")
        shared_model.load_state_dict(torch.load(args.weights))
        print("loaded model")
        results = mp.Array('f', len(training_objects))
        for rank, obj in enumerate(training_objects):
            p = mp.Process(target=test, args=(training_scene, obj, rank, shared_model, \
                            results, config, vars(args)))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("Testing accuracies:", list(zip(training_objects, results[:])))
        
