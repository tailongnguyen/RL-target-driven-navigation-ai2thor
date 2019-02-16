import ai2thor.controller
import numpy as np
import cv2
import h5py
import click
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

ALL_POSSIBLE_ACTIONS = [
	'MoveAhead',
	'MoveRight',
	'MoveBack',
	'MoveLeft',
]

def dump(scene="FloorPlan21", resolution=(300, 300)):

	f = h5py.File("env/dumped/{}.hdf5".format(scene), "w")

	observations = []

	locations = []
	visible_objects = []

	controller = ai2thor.controller.Controller()
	controller.start()

	controller.reset(scene)
	controller.random_initialize(unique_object_types=True)
	event = controller.step(dict(action='Initialize', gridSize=0.5))
	y_coord = event.metadata['agent']['position']['y']

	locations.append((event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z']))
	visited = set()
	visited.add((event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z']))

	while len(locations) > 0:
		loc = locations.pop(0)
		for act in ALL_POSSIBLE_ACTIONS:	
			controller.step(dict(action='Teleport', x=loc[0], y=y_coord, z=loc[1]))
			event = controller.step(dict(action=act))

			if event.metadata['lastActionSuccess']:
				new_loc = (event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'])
				if new_loc not in visited:
					visited.add(new_loc)
					locations.append(new_loc)

	all_locs = list(visited)
	states = []

	for loc in all_locs:
		for rot in [0, 90, 180, 270]:
			for horot in [-30, 0, 30, 60]:
				states.append((loc[0], loc[1], rot, horot))

	loc2idx = dict(zip(states, range(len(states))))
	graph = np.zeros(shape=(len(states), 6), dtype=np.float32)

	directions = {0: 1, 90: 1, 180: -1, 270: -1}

	for state in states:
		loc = (state[0], state[1])
		rot = state[2]
		horot = state[3]
		
		to_states = []

		if rot == 0 or rot == 180:
			to_states.append((loc[0], loc[1] + directions[rot] * 0.5, rot, horot)) # move ahead
			to_states.append((loc[0], loc[1] - directions[rot] * 0.5, rot, horot)) # move back

		else:
			to_states.append((loc[0] + directions[rot] * 0.5, loc[1], rot, horot)) # move ahead
			to_states.append((loc[0] - directions[rot] * 0.5, loc[1], rot, horot)) # move back

		to_states.append((loc[0], loc[1], rot + 90 if rot <= 180 else 0, horot)) # turn right
		to_states.append((loc[0], loc[1], rot - 90 if rot >= 90 else 270, horot)) # turn left
		
		to_states.append((loc[0], loc[1], rot, horot + 30)) # look down
		to_states.append((loc[0], loc[1], rot, horot - 30)) # look up

		state_idx = loc2idx[state]
		for i, new_state in enumerate(to_states):
			if new_state in loc2idx:
				graph[state_idx][i] = loc2idx[new_state]
			else:
				graph[state_idx][i] = -1

	for state in states:
		event = controller.step(dict(action='TeleportFull', x=state[0], y=y_coord, z=state[1], rotation=state[2], horizon=state[3]))

		resized_frame = cv2.resize(event.frame, (resolution[0], resolution[1]))
		observations.append(resized_frame)
		
		visible = list(np.unique([obj['objectType'] for obj in event.metadata['objects'] if obj['visible']]))

		if len(visible) > 0:
			visible_objects.append(",".join(visible))
		else:
			visible_objects.append("")

	print("{} states".format(len(states)))

	controller.stop()

	f.create_dataset("locations", data=np.asarray(states, np.float32))
	f.create_dataset("observations", data=np.asarray(observations, np.uint8))
	f.create_dataset("graph", data=graph)
	f.create_dataset("visible_objects", data=np.array(visible_objects, dtype=object), dtype=h5py.special_dtype(vlen=str))
	f.close()

def dump_resnet(scene="FloorPlan28"):

	tmp = models.resnet50(pretrained=True)
	modules = list(tmp.children())[:-1]
	extractor = nn.Sequential(*modules)
	extractor.cuda()

	f = h5py.File("env/dumped/{}.hdf5".format(scene), "a")
	observations = f['observations']
	resnet_features = []

	print("{} states in scene {}".format(observations.shape[0], scene))
	for idx, ob in enumerate(observations):
		print("Dumping resnet: {}/{}".format(idx + 1, observations.shape[0]), end='\r', flush=True)
		resized_frame = np.transpose(ob, (2, 0 ,1))

		inp = torch.from_numpy(resized_frame).unsqueeze(0).type(torch.cuda.FloatTensor)
		
		feature = extractor(inp)
		feature = feature.squeeze().detach().cpu().numpy()[:, np.newaxis]
		resnet_features.append(feature)

	f.create_dataset("resnet_features", data=np.asarray(resnet_features, np.float32))
	f.close()

if __name__ == '__main__':
	config = json.load(open("config.json"))
	kitchen_scenes = config['rooms']['Kitchens']['scenes']

	# for scene in kitchen_scenes[:1]:
	# 	print("Dumping {}".format(scene))
	# 	# dump(scene, config['resolution'])
	# 	dump_resnet(scene)

	scene = "FloorPlan28"
	dump(scene, config['resolution'])
	dump_resnet(scene)