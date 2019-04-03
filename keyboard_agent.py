import ai2thor.controller
import sys
import numpy as np
import h5py
import click
import json
import pyglet

from PIL import Image

ALL_POSSIBLE_ACTIONS = [
	'MoveAhead',
	'MoveBack',
	'RotateRight',
	'RotateLeft',
	# 'Stop'   
]

class SimpleImageViewer(object):

  def __init__(self, display=None):
    self.window = None
    self.isopen = False
    self.display = display

  def imshow(self, arr):
    if self.window is None:
      height, width, channels = arr.shape
      self.window = pyglet.window.Window(width=width, height=height, display=self.display, caption="THOR Browser")
      self.width = width
      self.height = height
      self.isopen = True

    assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
    image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    image.blit(0,0)
    self.window.flip()

  def close(self):
    if self.isopen:
      self.window.close()
      self.isopen = False

  def __del__(self):
    self.close()

def run(file_name=None):
	# file_name = file_path.split('/')[-1].split('.')[0]
	controller = ai2thor.controller.Controller()
	controller.start()

	controller.reset("FloorPlan1")
	event = controller.step(dict(action='Initialize', gridSize=0.5))
	y_coord = event.metadata['agent']['position']['y']
	all_visible_objects = list(np.unique([obj['objectType'] for obj in event.metadata['objects']]))
	print(y_coord)
	while True:  # making a loop
		try:  # used try so that if user pressed other than the given key error will not be shown
			key = click.getchar()
			if key =='a':  # Rotate Left
				event = controller.step(dict(action='RotateLeft'))           
			elif key =='d':
				event = controller.step(dict(action='RotateRight'))
			elif key =='w':
				event = controller.step(dict(action='MoveAhead'))
			elif key =='s':
				event = controller.step(dict(action='MoveBack'))
			elif key =='z':
				event = controller.step(dict(action='LookDown'))
			elif key =='x':
				event = controller.step(dict(action='LookUp'))
			elif key =='q':
				controller.stop()
				break
			elif key =='r':
				scene = input("Scene id: ")
				controller.reset('FloorPlan{}'.format(scene))
				controller.random_initialize(unique_object_types=True)
				event = controller.step(dict(action='Initialize', gridSize=0.5))
			else:
				print("Key not supported! Try a, d, w, s, q, r.")
			print((event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'], event.metadata['agent']['rotation']))
			print([(obj['objectType'], obj['distance']) for obj in event.metadata['objects'] if obj['visible']])
		except:
			print("Key not supported! Try a, d, w, s, q, r.")


def key_press(key, mod):
	global human_agent_action, human_wants_restart, stop_requested
	if key == ord('R') or key == ord('r'): # r/R
		human_wants_restart = True
	if key == ord('Q') or key == ord('q'): # q/Q
		stop_requested = True

	if key == 0xFF52: # move ahead
		human_agent_action = 0
	if key == 0xFF54: # move back
		human_agent_action = 1
	if key == 0xFF53: # turn right
		human_agent_action = 2
	if key == 0xFF51: # turn left
		human_agent_action = 3

	if key == ord('z'): # look down
		human_agent_action = 4
	if key == ord('x'): # look up
		human_agent_action = 5

if __name__ == '__main__':
	
	# run()

	human_agent_action = None
	human_wants_restart = False
	stop_requested = False
	next_position = None
	visible = None

	f = h5py.File('dumped/FloorPlan10.hdf5', "r")
	observations = f['observations']
	graph = f['graph']
	visible_objects = f['visible_objects']
	dump_features = f['dump_features']

	config = json.load(open('config.json'))
	categories = list(config['new_objects'].keys())

	current_position = np.random.randint(0, observations.shape[0])

	viewer = SimpleImageViewer()
	viewer.imshow(observations[current_position].astype(np.uint8))
	viewer.window.on_key_press = key_press

	print("Use arrow keys to move the agent.")
	print("Press R to reset agent\'s location.")
	print("Press Q to quit.")

	while True:
		# waiting for keyboard input
		if human_agent_action is not None:
			# move actions
			next_position = graph[current_position][human_agent_action]
			current_position = next_position if next_position != -1 else current_position
			print(dump_features[current_position][-4:])
			distances = [(categories[i], dump_features[current_position][i]) for i in list(np.where(dump_features[current_position][:-4] > 0)[0])]
			visible = visible_objects[current_position].split(',')
			human_agent_action = None

		# waiting for reset command
		if human_wants_restart:
			# reset agent to random location
			current_position = np.random.randint(0, observations.shape[0])
			human_wants_restart = False

		# check collision
		if next_position == -1:
			print('Collision occurs.')

		# check quit command
		if stop_requested: break

		viewer.imshow(observations[current_position].astype(np.uint8))
		if visible is not None and len(list(visible)) > 0:
			print("Visible: {}".format(visible))
			visible = None

	print("Goodbye.")