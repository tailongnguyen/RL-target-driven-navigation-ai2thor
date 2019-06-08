import numpy as np
import gym
import cv2
import h5py
import os
import sys
import random

class AI2ThorDumpEnv():
    """
    Wrapper base class
    """
    def __init__(self, scene, target, config, arguments=dict(), seed=None):
        """
        :param seed: (int)   Random seed
        :param config: (str)   Dictionary file storing cofigurations
        :param: scene: (list)  Scene to train
        :param: target: (list)  Target object to train
        """
        if seed is not None:
            np.random.seed(seed)

        self.config = config
        self.arguments = arguments
        self.scene = scene
        self.target = target
        self.history_size = arguments.get('history_size')
        self.action_size = arguments.get('action_size')

        self.h5_file = h5py.File("{}.hdf5".format(os.path.join(config['dump_path'], self.scene)), 'r')

        self.states = self.h5_file['locations'][()]
        self.graph = self.h5_file['graph'][()]
        self.scores = self.h5_file['resnet_scores'][()] if not arguments['yolo_gcn'] else self.h5_file['dump_features'][()][:, :-4].astype(bool).astype(int)
        self.all_visible_objects = self.h5_file['all_visible_objects'][()].tolist()
        self.visible_objects = self.h5_file['visible_objects'][()]
        self.observations = self.h5_file['observations'][()]

        assert self.target in self.all_visible_objects, "Target {} is unreachable in {}!".format(self.target, self.scene)

        self.resnet_features = self.h5_file['resnet_features'][()]
        self.dump_features = self.h5_file['dump_features'][()]

        if arguments['onehot']:
            self.features = self.dump_features
        else:
            self.features = self.resnet_features

        assert self.action_size <= self.graph.shape[1], "The number of actions exceeds the limit of environment."

        if "shortest" in self.h5_file.keys():
            self.shortest = self.h5_file['shortest'][()]

        if self.arguments['hard']:
            # agent has to reach the correct position and has right rotation
            self.offset = 3
        else:
            # agent only has to reach the correct position
            self.offset = 2

        self.target_ids = [idx for idx in range(len(self.states)) if self.target in self.visible_objects[idx].split(",")]
        self.target_locs = set([tuple(self.states[idx][:self.offset]) for idx in self.target_ids])        

        self.action_space = self.action_size 
        self.cv_action_onehot = np.identity(self.action_space)
        
        self.history_states = np.zeros((self.history_size, self.features.shape[1]))
        self.observations_stack = [np.zeros((3, 128, 128)) for _ in range(self.history_size)]

    def step(self, action):
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
            if action == 2 or action == 3:
                for _ in range(int(self.arguments['angle'] / 22.5)):
                    self.current_state_id = int(self.graph[k][action])
            else:                
                self.current_state_id = int(self.graph[k][action])
                
            if tuple(self.states[self.current_state_id][:self.offset]) in self.target_locs:
                self.terminal = True
                self.collided = False
            else:
                self.terminal = False
                self.collided = False
        else:
            self.terminal = False
            self.collided = True

        reward, done = self.transition_reward()

        self.update_states()

        if self.arguments['train_cnn']:
            return np.asarray(self.observations_stack, dtype=np.float32), self.scores[self.current_state_id], reward, done
        else:
            return self.history_states, self.scores[self.current_state_id], reward, done

    def transition_reward(self):
        reward = self.config['default_reward']
        done = 0
        if self.terminal:
            reward = self.config['success_reward']
            done = 1
        elif self.arguments['anti_col'] and self.collided:
            reward = self.config['collide_reward']

        return reward, done

    def reset(self):
        # reset parameters
        k = random.randrange(self.states.shape[0])

        while self.states[k][2] % self.arguments['angle'] != 0.0:
            k = random.randrange(self.states.shape[0])

        self.current_state_id = k

        self.update_states(reset=True)
        self.terminal = False
        self.collided = False

        if self.arguments['train_cnn']:
            return np.asarray(self.observations_stack, dtype=np.float32), self.scores[self.current_state_id], self.target
        else:
            return self.history_states, self.scores[self.current_state_id], self.target

    def update_states(self, reset=False): 
        if reset:
            self.history_states = np.zeros((self.history_size, self.features.shape[1]))
            self.observations_stack = [np.zeros((3, 128, 128)) for _ in range(self.history_size)]

        f = self.features[self.current_state_id]
            
        self.history_states = np.append(self.history_states[1:, :], f[np.newaxis, :], 0)

        self.observations_stack.append(self.observation())
        self.observations_stack = self.observations_stack[1:]

    def state(self):    
        return self.features[self.current_state_id]

    def observation(self):
        ob = self.observations[self.current_state_id]        
        resized_ob = cv2.resize(ob, (128, 128))
        return np.transpose(resized_ob, (2, 0, 1))

class MultiSceneEnv():
    """
    Wrapper base class
    """
    def __init__(self, scene, config, arguments=dict(), seed=None):
        """
        :param seed: (int)   Random seed
        :param config: (str)   Dictionary file storing cofigurations
        :param: scene: (list)  Scene to train
        :param: objects: (list)  Target objects to train
        """

        if seed is not None:
            np.random.seed(seed)
            
        self.config = config
        self.arguments = arguments
        self.scene = scene

        self.history_size = arguments.get('history_size')
        self.action_size = arguments.get('action_size')

        scene_id = int(scene.split("FloorPlan")[1])
        if scene_id > 0 and scene_id < 31:
            room_type = "Kitchens"
        elif scene_id > 200 and scene_id < 231:
            room_type = 'Living Rooms'
        elif scene_id > 300 and scene_id < 331:
            room_type = 'Bedrooms'
        elif scene_id > 400 and scene_id < 431:
            room_type = 'Bathrooms'
        else:
            raise KeyError

        if arguments['test'] == 1:
            self.targets = config["rooms"][room_type]['train_objects'] + config["rooms"][room_type]['test_objects']
        else:
            self.targets = config["rooms"][room_type]['train_objects']

        self.h5_file = h5py.File("{}.hdf5".format(os.path.join(config['dump_path'], self.scene)), 'r')

        self.states = self.h5_file['locations'][()]
        self.graph = self.h5_file['graph'][()]
        self.scores = self.h5_file['resnet_scores'][()] if not arguments['yolo_gcn'] else self.h5_file['dump_features'][()][:, :-4].astype(bool).astype(int)
        self.all_visible_objects = self.h5_file['all_visible_objects'][()].tolist()
        self.visible_objects = self.h5_file['visible_objects'][()]
        self.observations = self.h5_file['observations'][()]

        self.resnet_features = self.h5_file['resnet_features'][()]
        self.dump_features = self.h5_file['dump_features'][()]

        
        if arguments['onehot']:
            self.features = self.dump_features
        else:
            self.features = self.resnet_features

        assert self.action_size <= self.graph.shape[1], "The number of actions exceeds the limit of environment."

        if "shortest" in self.h5_file.keys():
            self.shortest = self.h5_file['shortest'][()]

        if self.arguments['hard']:
            # agent has to reach the correct position and has right rotation
            self.offset = 3
        else:
            # agent only has to reach the correct position
            self.offset = 2

        self.target = np.random.choice(self.targets)
        self.target_ids = [idx for idx in range(len(self.states)) if self.target in self.visible_objects[idx].split(",")]
        self.target_locs = set([tuple(self.states[idx][:self.offset]) for idx in self.target_ids])        
        
        self.action_space = self.action_size 
        self.cv_action_onehot = np.identity(self.action_space)
        
        self.history_states = np.zeros((self.history_size, self.features.shape[1]))
        self.observations_stack = [np.zeros((3, 128, 128)) for _ in range(self.history_size)]


    def step(self, action):
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
            if action == 2 or action == 3:
                for _ in range(int(self.arguments['angle'] / 22.5)):
                    self.current_state_id = int(self.graph[k][action])
            else:                
                self.current_state_id = int(self.graph[k][action])

            if tuple(self.states[self.current_state_id][:self.offset]) in self.target_locs:
                self.terminal = True
                self.collided = False
            else:
                self.terminal = False
                self.collided = False
        else:
            self.terminal = False
            self.collided = True

        reward, done = self.transition_reward()

        self.update_states()

        if self.arguments['train_cnn']:
            return np.asarray(self.observations_stack, dtype=np.float32), self.scores[self.current_state_id], reward, done
        else:
            return self.history_states, self.scores[self.current_state_id], reward, done

    def transition_reward(self):
        reward = self.config['default_reward']
        done = 0
        if self.terminal:
            reward = self.config['success_reward']
            done = 1
        elif self.arguments['anti_col'] and self.collided:
            reward = self.config['collide_reward']

        return reward, done

    def reset(self):
        self.target = np.random.choice(self.targets)
        self.target_ids = [idx for idx in range(len(self.states)) if self.target in self.visible_objects[idx].split(",")]
        self.target_locs = set([tuple(self.states[idx][:self.offset]) for idx in self.target_ids])        

        k = random.randrange(self.states.shape[0])

        while self.states[k][2] % self.arguments['angle'] != 0.0:
            k = random.randrange(self.states.shape[0])

        # reset parameters
        self.current_state_id = k
        
        self.update_states(reset=True)
        self.terminal = False
        self.collided = False

        if self.arguments['train_cnn']:
            return np.asarray(self.observations_stack, dtype=np.float32), self.scores[self.current_state_id], self.target
        else:
            return self.history_states, self.scores[self.current_state_id], self.target


    def update_states(self, reset=False): 
        if reset:
            self.history_states = np.zeros((self.history_size, self.features.shape[1]))
            self.observations_stack = [np.zeros((3, 128, 128)) for _ in range(self.history_size)]

        f = self.features[self.current_state_id]
            
        self.history_states = np.append(self.history_states[1:, :], f[np.newaxis, :], 0)

        self.observations_stack.append(self.observation())
        self.observations_stack = self.observations_stack[1:]

    def state(self):    
        return self.features[self.current_state_id]

    def observation(self):
        ob = self.observations[self.current_state_id]        
        resized_ob = cv2.resize(ob, (128, 128))
        return np.transpose(resized_ob, (2, 0, 1))


if __name__ == '__main__':
    AI2ThorEnv()
