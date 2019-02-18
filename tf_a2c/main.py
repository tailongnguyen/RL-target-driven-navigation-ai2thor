import os 
import argparse
import sys
import h5py
import json

from train import MultitaskPolicy

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

	# np.random.shuffle(all_scenes)
	# training_scenes = all_scenes[:20]
	# validation_scenes = all_scenes[20:25]
	# testing_scenes = all_scenes[25:]

	training_scene = all_scenes[arguments['scene_id']]

	# h5_file = h5py.File("{}.hdf5".format(os.path.join(config['dump_path'], training_scene)), 'r')
	# all_visible_objects = set(",".join([o for o in list(h5_file['visible_objects']) if o != '']).split(','))
	# trainable_objects = list(set(train_objects).intersection(all_visible_objects))
	# h5_file.close()

	trainable_objects = {
		0: ['Cabinet', 'Knife', 'CoffeeMachine', 'StoveKnob', 'StoveBurner', 'ButterKnife', 'Sink', 'Fridge', 'TableTop'],
		27: ['Cabinet', 'TableTop', 'StoveKnob', 'Fridge', 'Sink', 'StoveBurner', 'CoffeeMachine']
	}

	# print(trainable_objects)
	# trainable_objects = ['CoffeeMachine', 'Sink']
	training_object = trainable_objects[arguments['scene_id']][arguments['target_id']]

	print("Training scene: {} | Target: {}".format(training_scene, [training_object]))
	multitask_agent = MultitaskPolicy(training_scene, [training_object], config, arguments)
	multitask_agent.train()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('--num_task', nargs='?', type=int, default=1, 
						help='Number of tasks to train on')
	parser.add_argument('--share_latent', nargs='?', type=int, default=0,
						help='Whether to join the latent spaces of actor and critic')
	parser.add_argument('--num_episodes', nargs='?', type=int, default=16,
						help='Number of episodes to sample in each epoch')
	parser.add_argument('--num_iters', nargs='?', type=int, default=100,
						help='Number of steps to be sampled in each episode')
	parser.add_argument('--lr', nargs='?', type=float, default=1e-3,
						help='Learning rate')
	parser.add_argument('--use_gae', nargs='?', type=int, default=1,
						help='Whether to use generalized advantage estimate')
	parser.add_argument('--num_epochs', nargs='?', type=int, default=15000,
						help='Number of epochs to train')
	parser.add_argument('--gamma', nargs='?', type=float, default=0.99,
						help='Coeff for return estimation')
	parser.add_argument('--lamb', nargs='?', type=float, default=0.96,
						help='Coeff for GAE estimation')
	parser.add_argument('--ec', nargs='?', type=float, default=0.01,
						help='Entropy coeff in total loss')
	parser.add_argument('--vc', nargs='?', type=float, default=0.5,
						help='Value loss coeff in total loss')
	parser.add_argument('--max_gradient_norm', nargs='?', type=float, default=None,
						help='')
	parser.add_argument('--train-resnet', type=int, default=0,
						help='whether to include resnet into training')
	parser.add_argument('--history-size', type=int, default=1,
						help='whether to include resnet into training')
	parser.add_argument('--decay', nargs='?', type=int, default=1,
						help='Whether to decay the learning_rate')
	parser.add_argument('--noise_argmax', nargs='?', type=int, default=1,
						help='Whether touse noise argmax in action sampling')
	parser.add_argument('--joint_loss', nargs='?', type=int, default=0,
						help='Whether to join loss function')
	parser.add_argument('--room_id', type=int, default=0,
						help='room id (default: 0)')
	parser.add_argument('--scene_id', type=int, default=0,
						help='scene id (default: 0)')
	parser.add_argument('--target_id', type=int, default=0,
						help='target id (default: 0)')
	parser.add_argument('--logging', type=str, default="training-history/",
						help='Logging folder')
	parser.add_argument('--config_file', type=str, default="config.json")

	args = parser.parse_args()

	# print(vars(args))
	config = read_config(args.config_file)
	main(config, vars(args))
