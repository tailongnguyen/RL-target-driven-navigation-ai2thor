from env.ai2thor_env import AI2ThorDumpEnv
from agent import Agent
from utils import LinearSchedule
from datetime import datetime

import tensorflow as tf
import numpy as np
import os
import random
import time
import json
import argparse

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

def main(config, arguments):
    room = config['rooms'][ALL_ROOMS[arguments['room_id']]]
    all_scenes = room['scenes']
    train_objects = room['train_objects']
    test_objects = room['test_objects']

    training_scene = all_scenes[arguments['scene_id']]

    # h5_file = h5py.File("{}.hdf5".format(os.path.join(config['dump_path'], training_scene)), 'r')
    # all_visible_objects = set(",".join([o for o in list(h5_file['visible_objects']) if o != '']).split(','))
    # print(all_visible_objects)
    # trainable_objects = list(set(train_objects).intersection(all_visible_objects))
    # h5_file.close()
    # print(trainable_objects)

    trainable_objects = {
        0: ['Knife', 'Sink', 'CoffeeMachine', 'StoveKnob', 'StoveBurner', 'Cabinet', 'Fridge', 'TableTop'],
        1: ['CoffeeMachine', 'StoveBurner', 'Sink', 'GarbageCan', 'TableTop', 'Fridge',  'Mug', 'StoveKnob', 'Microwave', 'Cabinet', 'Chair'],
        27: ['Cabinet', 'TableTop', 'StoveKnob', 'Fridge', 'Sink', 'StoveBurner', 'CoffeeMachine']
    }

    training_object = trainable_objects[arguments['scene_id']][arguments['target_id']]

    env = AI2ThorDumpEnv(training_scene, training_object, config, arguments)

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=arguments['gpu_fraction'])
    sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))

    agent = Agent(sess, env.features.shape[1], env.action_space, int(time.time() * 100) % 100, arguments)
    sess.run(tf.global_variables_initializer())


    saver = tf.train.Saver()
    timer = "{}_{}_{}".format(str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-"), training_scene, training_object)
    log_folder = os.path.join(arguments.get('logging'), timer)
    writer = tf.summary.FileWriter(log_folder)

    reward_log = tf.placeholder(tf.float32)
    redundant_log = tf.placeholder(tf.float32)

    test_name =  training_scene
    tf.summary.scalar(test_name + "/" + training_object + "/rewards", reward_log)
    tf.summary.scalar(test_name + "/" + training_object + "/redundants", redundant_log)

    write_op = tf.summary.merge_all()

    num_epochs = arguments['num_epochs']
    num_steps = arguments['num_iters']

    epsilon_schedule = LinearSchedule(num_epochs, final_p=0.02)
    ep_rewards = []
    start_time = time.time()
    for ep in range(num_epochs):
        state, target = env.reset()
        start = env.current_state_id
        rewards = 0
        redundant = 0

        for step in range(num_steps):
            action = agent.act(state, epsilon_schedule.value(ep))
            next_state, reward, done = env.step(action)
            agent.step(state, env.cv_action_onehot[action], reward, next_state, done)
            state = next_state

            rewards += reward
            if done: 
                break

        if not done:
            end = env.current_state_id
            try:
                redundants = []
                for target_id in env.target_ids:
                    redundants.append(num_steps + env.shortest[end, target_id] - env.shortest[start, target_id])

                redundant = min(redundants)
            except AttributeError:
                pass

        ep_rewards.append(rewards)
        print("Ep {}/{}, elapsed time: {:.3f} | rewards: {:.3f}| mean rewards: {:.3f}".format(
                ep+1, num_epochs, (time.time() - start_time)/3600, 
                rewards, np.mean(ep_rewards)), end='\r', flush=True)
        if ep % 100 == 0:
            print("Ep {}/{}, elapsed time: {:.3f} | rewards: {:.3f}| mean rewards: {:.3f}\n".format(
                ep+1, num_epochs, (time.time() - start_time)/3600, 
                rewards, np.mean(ep_rewards)))

        summary = sess.run(write_op, feed_dict = {
            reward_log: rewards,
            redundant_log: redundant,   
            })

        writer.add_summary(summary, ep + 1)
        writer.flush()
    
    saver.save(sess, log_folder + "/my-model")
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--room_id', type=int, default=0)
    parser.add_argument('--scene_id', nargs='?', type=int, default=0)
    parser.add_argument('--target_id', nargs='?', type=int, default=0)
    parser.add_argument('--gpu_fraction', nargs='?', type=float, default=0.15,
                        help='GPU memory usage fraction')
    parser.add_argument('--history_size', type=int, default=1,
                        help='whether to stack frames to make input')
    parser.add_argument('--num_epochs', nargs='?', type=int, default=10000,
                        help='Number of epochs to train')
    parser.add_argument('--num_iters', nargs='?', type=int, default=100,
                        help='Number of steps to be sampled in each episode')
    parser.add_argument('--buffer_size', nargs='?', type=int, default=100000,
                        help='replay buffer size')
    parser.add_argument('--batch_size', nargs='?', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--gamma', nargs='?', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--tau', nargs='?', type=float, default=1e-3,
                        help='for soft update of target parameters')
    parser.add_argument('--lr', nargs='?', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--update_every', nargs='?', type=int, default=4,
                        help='how often to update the network')
    parser.add_argument('--logging', type=str, default="training-history/",
                        help='Logging folder')
    parser.add_argument('--config_file', type=str, default="config.json")


    args = parser.parse_args()

    # print(vars(args))
    config = read_config(args.config_file)
    main(config, vars(args))
