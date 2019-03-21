import os 
import argparse
import sys
import h5py
import json
import multiprocessing

from single_task import SingleTaskPolicy
from multi_task import MultiTaskPolicy
from sharing_polices import SharingPolicy

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

def worker(training_scene, training_object, config, arguments):
    print("Training scene: {} | Target: {}".format(training_scene, training_object))
    agent = SingleTaskPolicy(training_scene, training_object, config, arguments)
    agent.train()

def main(config, arguments):
    training_scene = "FloorPlan{}".format(arguments['scene_id'])
    trainable_objects = config["picked"][training_scene]

    if arguments['mode'] == 0:
        worker(training_scene, trainable_objects['train'][arguments['target_id']], config, arguments)
    else:
        trainable_objects = trainable_objects['train']

        print(list(zip(range(len(trainable_objects)), trainable_objects)))

        command = input("Please specify targets: ")

        if '-' not in command:
            target_ids = [int(i.strip()) for i in command.split(",")]
        else:
            target_ids = list(range(int(command.split('-')[0]), int(command.split('-')[1]) + 1))

        trainable_objects = [trainable_objects[target_id] for target_id in target_ids]
        print("Training scene: {} | Target: {}".format(training_scene, trainable_objects))


        if arguments['mode'] == 1:
            print("Starting {} processes ..".format(len(trainable_objects)))

            processes = []
            for target in trainable_objects:
                p = multiprocessing.Process(target=worker, args=(training_scene, target, config, arguments))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

        elif arguments['mode'] == 2:
            
            agent = MultiTaskPolicy(training_scene, trainable_objects, config, arguments)
            agent.train()

        elif arguments['mode'] == 3:
            assert len(trainable_objects) == 2, "> 3 sharing is not supported."
        
            agents = SharingPolicy(training_scene, trainable_objects, config, arguments)
            agents.train()

        else:
            import sys
            sys.exit("Invalid mode.")

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--mode', nargs='?', type=int, default=0,
                        help='Running mode. 0: run one single task. \
                                1: run multiple task in parallel.\
                                2: run a multitask agent\
                                3: run sharing-exp agents')
    parser.add_argument('--share_latent', nargs='?', type=int, default=0,
                        help='Whether to join the latent spaces of actor and critic')
    parser.add_argument('--num_episodes', nargs='?', type=int, default=16,
                        help='Number of episodes to sample in each epoch')
    parser.add_argument('--num_iters', nargs='?', type=int, default=100,
                        help='Number of steps to be sampled in each episode')
    parser.add_argument('--gpu_fraction', nargs='?', type=float, default=0.15,
                        help='GPU memory usage fraction')
    parser.add_argument('--lr', nargs='?', type=float, default=7e-4,
                        help='Learning rate')
    parser.add_argument('--use_gae', nargs='?', type=int, default=1,
                        help='Whether to use generalized advantage estimate')
    parser.add_argument('--embed', nargs='?', type=int, default=1,
                        help='Whether to use text embedding for multitask')
    parser.add_argument('--num_epochs', nargs='?', type=int, default=10000,
                        help='Number of epochs to train')
    parser.add_argument('--gamma', nargs='?', type=float, default=0.99,
                        help='Coeff for return estimation')
    parser.add_argument('--lamb', nargs='?', type=float, default=0.96,
                        help='Coeff for GAE estimation')
    parser.add_argument('--ec', nargs='?', type=float, default=0.01,
                        help='Entropy coeff in total loss')
    parser.add_argument('--vc', nargs='?', type=float, default=0.5,
                        help='Value loss coeff in total loss')
    parser.add_argument('--dropout', nargs='?', type=float, default=-1,
                        help='Value loss coeff in total loss')
    parser.add_argument('--max_gradient_norm', nargs='?', type=float, default=50,
                        help='')
    parser.add_argument('--anti_col', type=int, default=0,
                        help='whether to include collision penalty to rewarding scheme')
    parser.add_argument('--train_resnet', type=int, default=0,
                        help='whether to include resnet into training')
    parser.add_argument('--history_size', type=int, default=4,
                        help='number of frames to be stacked as input')
    parser.add_argument('--action_size', type=int, default=4,
                        help='number of possible actions')
    parser.add_argument('--decay', nargs='?', type=int, default=1,
                        help='Whether to decay the learning_rate')
    parser.add_argument('--noise_argmax', nargs='?', type=int, default=1,
                        help='Whether touse noise argmax in action sampling')
    parser.add_argument('--joint_loss', nargs='?', type=int, default=0,
                        help='Whether to join loss function')
    parser.add_argument('--room_id', type=int, default=0,
                        help='room id (default: 0)')
    parser.add_argument('--scene_id', type=int, default=1,
                        help='scene id (default: 0)')
    parser.add_argument('--target_id', type=int, default=0,
                        help='target id (default: 0)')
    parser.add_argument('--logging', type=str, default="training-history/",
                        help='Logging folder')
    parser.add_argument('--config_file', type=str, default="../config.json")

    args = parser.parse_args()

    # print(vars(args))
    config = read_config(args.config_file)
    main(config, vars(args))
