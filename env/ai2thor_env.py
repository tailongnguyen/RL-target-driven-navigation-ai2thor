import ai2thor.controller
import numpy as np
import gym
import cv2
import h5py
import os
import sys
import random

from skimage import transform
from copy import deepcopy
from gym import error, spaces
from gym.utils import seeding

ALL_POSSIBLE_ACTIONS = [
    'MoveAhead',
    'MoveBack',
    'RotateRight',
    'RotateLeft',
    # 'Stop'   
]


class AI2ThorEnv(gym.Env):
    """
    Wrapper base class
    """
    def __init__(self, config, scenes, objects, seed=None):
        """
        :param seed: (int)   Random seed
        :param config: (str)   Path to environment configuration file. Either absolute or
                                     relative path to the root of this repository.
        :param: scenes: (list)  List of scene ids to train on
        :param: objects: (list)  List of target objects to train on
        """
        
        self.config = config
        self.scenes = scenes
        self.objects = objects
        self.scene_id = np.random.choice(self.scenes)
        self.target = np.random.choice(self.objects)

        # Randomness settings
        self.np_random = None
        if seed:
            self.seed(seed)
        
        # Action settings
        self.action_names = tuple(ALL_POSSIBLE_ACTIONS.copy())
        
        self.action_space = spaces.Discrete(len(self.action_names))
        # Image settings
        self.event = None
        channels = 1 if self.config['grayscale'] else 3
        self.observation_space = spaces.Box(low=0, 
                                            high=255, 
                                            shape=(channels, self.config['resolution'][0],
                                            self.config['resolution'][1]),
                                            dtype=np.uint8)

        self.history_frames = [np.zeros(self.observation_space.shape) for i in range(self.config['history_size'])]

        # Start ai2thor
        self.controller = ai2thor.controller.Controller()
        self.controller.start()

    def step(self, action, verbose=True):
        if not self.action_space.contains(action):
            raise error.InvalidAction('Action must be an integer between '
                                      '0 and {}!'.format(self.action_space.n))
        action_str = self.action_names[action]

        # Move or Rotate actions
        self.event = self.controller.step(dict(action=action_str))

        state_image = self.preprocess(self.event.frame)
        self.history_frames += [state_image]
        self.history_frames = self.history_frames[1:]

        reward, done = self.transition_reward()

        return self.history_frames, reward, done, self.target

    def transition_reward(self):
        visible_objects = [obj['objectType'] for obj in self.event.metadata['objects'] if obj['visible']]
        reward = -0.01
        done = 0
        if self.target in visible_objects:
            reward = 10.0
            done = 1

        return reward, done

    def preprocess(self, img):
        """
        Compute image operations to generate state representation
        """
        img = cv2.resize(img, tuple(self.config['resolution']))
        img = img.astype(np.float32)
        if self.observation_space.shape[0] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.transpose(img, axes=(2,0,1))
        print("shape:", img.shape)
        return img

    def reset(self):
        print('Resetting environment and starting new episode')
        self.scene_id = np.random.choice(self.scenes)
        self.target = np.random.choice(self.objects)
        print("Looking for {} in {}".format(self.target, self.scene_id))
        
        self.controller.reset(self.scene_id)
        self.event = self.controller.step(dict(action='Initialize', gridSize=0.25,
                                               renderDepthImage=True, renderClassImage=True,
                                               renderObjectImage=True))
        state = self.preprocess(self.event.frame)
        self.history_frames[-1] = state
        return self.history_frames, self.target

    def render(self, mode='human'):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        return seed1

    def close(self):
        self.controller.stop()

class AI2ThorDumpEnv():
    """
    Wrapper base class
    """
    def __init__(self, config, scenes, objects, seed=None):
        """
        :param seed: (int)   Random seed
        :param config: (str)   Path to environment configuration file. Either absolute or
                                     relative path to the root of this repository.
        :param: scenes: (list)  List of scene ids to train on
        :param: objects: (list)  List of target objects to train on
        """
        
        self.config = config
        self.scenes = scenes
        self.objects = objects
        
        tried = 0
        while 1:
            self.scene_id = np.random.choice(self.scenes)
            self.h5_file = h5py.File("{}.hdf5".format(os.path.join(config['dump_path'], self.scene_id)), 'r')

            all_visible_objects = set(",".join([o for o in list(self.h5_file['visible_objects']) if o != '']).split(','))
            self.objects = list(set(self.objects).intersection(all_visible_objects))
            if len(self.objects) > 0:
                break
            else:
                tried += 1
                self.h5_file.close()

                if tried >= 20:
                    sys.exit("Something went wrong in choosing target!")

        self.states = self.h5_file['locations'][()]
        self.graph = self.h5_file['graph'][()]
        self.features = self.h5_file['resnet_features'][()]
        self.visible_objects = self.h5_file['visible_objects'][()]

        self.target = np.random.choice(self.objects)
        self.target_ids = [idx for idx in range(len(self.states)) if self.target in self.visible_objects[idx].split(",")]

        self.action_space = self.graph.shape[1]
        
        # Randomness settings
        self.np_random = None
        if seed:
            self.seed(seed)
        
    
        self.history_states = np.zeros((self.features.shape[1], self.config['history_size']))

    def step(self, action, verbose=True):
        '''
        0: move ahead
        1: move back
        2: rotate right
        3: rotate left
        4: look down
        5: look up
        '''

        if action >= self.action_space:
            raise error.InvalidAction('Action must be an integer between '
                                      '0 and {}!'.format(self.action_space - 1))
        k = self.current_state_id
        if self.graph[k][action] != -1:
            self.current_state_id = int(self.graph[k][action])
            if self.current_state_id in self.target_ids:
                self.terminal = True
                collided = False
            else:
                self.terminal = False
                collided = False
        else:
            self.terminal = False
            collided = True

        reward, done = self.transition_reward(collided)

        self.tiled_states()

        return self.history_states.T, reward, done

    def transition_reward(self, collided):
        reward = -0.01
        done = 0
        if self.terminal:
            reward = 10.0
            done = 1
        elif self.config['anti-collision'] and collided:
            reward = -0.1

        return reward, done

    def reset(self):
        # reset parameters
        self.current_state_id = random.randrange(self.states.shape[0])
        self.tiled_states()
        self.terminal = False

        return self.history_states.T, self.target

    def tiled_states(self):
        f = self.features[self.current_state_id]
        self.history_states = np.append(self.history_states[:, 1:], f, axis = 1)

        assert self.history_states.shape == (self.features.shape[1], self.config['history_size']), \
        "something wrong with states buffering"

    def render(self, mode='human'):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        return seed1

if __name__ == '__main__':
    AI2ThorEnv()
