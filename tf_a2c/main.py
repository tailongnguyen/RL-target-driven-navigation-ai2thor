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

def main(config, custom_config):

	room = config['rooms'][ALL_ROOMS[custom_config['room_id']]]
	all_scenes = room['scenes']
	train_objects = room['train_objects']
	test_objects = room['test_objects']

	# np.random.shuffle(all_scenes)
	# training_scenes = all_scenes[:20]
	# validation_scenes = all_scenes[20:25]
	# testing_scenes = all_scenes[25:]

	training_scene = all_scenes[0]
	h5_file = h5py.File("{}.hdf5".format(os.path.join(config['dump_path'], training_scene)), 'r')
	all_visible_objects = set(",".join([o for o in list(h5_file['visible_objects']) if o != '']).split(','))
	trainable_objects = list(set(train_objects).intersection(all_visible_objects))
	h5_file.close()

	trainable_objects = ['CoffeeMachine', 'Sink']
	print("Training scene: {} | Targets: {}".format(training_scene, trainable_objects))
	multitask_agent = MultitaskPolicy(training_scene, trainable_objects, config, custom_config)
	multitask_agent.train()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('--num_task', nargs='?', type=int, default = None, 
						help='Number of tasks to train on')
	parser.add_argument('--share_latent', nargs='?', type=int, default = None,
						help='Whether to join the latent spaces of actor and critic')
	parser.add_argument('--num_episodes', nargs='?', type=int, default = None,
						help='Number of episodes to sample in each epoch')
	parser.add_argument('--num_iters', nargs='?', type=int, default = None,
						help='Number of steps to be sampled in each episode')
	parser.add_argument('--lr', nargs='?', type=float, default = None,
						help='Learning rate')
	parser.add_argument('--use_gae', nargs='?', type=int, default = None,
						help='Whether to use generalized advantage estimate')
	parser.add_argument('--num_epochs', nargs='?', type=int, default = None,
						help='Number of epochs to train')
	parser.add_argument('--gamma', nargs='?', type=float, default = None,
						help='Coeff for return estimation')
	parser.add_argument('--lamb', nargs='?', type=float, default = None,
						help='Coeff for GAE estimation')
	parser.add_argument('--ec', nargs='?', type=float, default = None,
						help='Entropy coeff in total loss')
	parser.add_argument('--vc', nargs='?', type=float, default = None,
						help='Value loss coeff in total loss')
	parser.add_argument('--max_gradient_norm', nargs='?', type=float, default = None,
						help='')
	parser.add_argument('--train-resnet', type=int, default=None,
						help='whether to include resnet into training')
	parser.add_argument('--history-size', type=int, default=None,
						help='whether to include resnet into training')
	parser.add_argument('--decay', nargs='?', type=int, default = None,
						help='Whether to decay the learning_rate')
	parser.add_argument('--noise_argmax', nargs='?', type=int, default = None,
						help='Whether touse noise argmax in action sampling')
	parser.add_argument('--joint_loss', nargs='?', type=int, default = None,
						help='Whether to join loss function')
	parser.add_argument('--room_id', type=int, default=0,
						help='room id (default: 0)')
	parser.add_argument('--logging', type=str, default=None,
						help='Logging folder')
	parser.add_argument('--config_file', type=str, default="config.json")

	args = parser.parse_args()

	# print(vars(args))
	config = read_config(args.config_file)
	main(config, vars(args))
