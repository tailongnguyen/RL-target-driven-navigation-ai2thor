import ai2thor.controller
import numpy as np
import cv2
import h5py
import re
import click
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

ALL_POSSIBLE_ACTIONS = [
    'MoveAhead',
    'MoveRight',
    'MoveBack',
    'MoveLeft',
]

def check_size(scene):
    f = h5py.File("dumped/{}.hdf5".format(scene), "w")

    locations = []
    visible_objects = []

    controller = ai2thor.controller.Controller()
    controller.start()

    controller.reset(scene)
    controller.random_initialize(unique_object_types=True)
    event = controller.step(dict(action='Initialize', gridSize=0.5))
    y_coord = event.metadata['agent']['position']['y']

    locations.append((event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z']))

    # Using BFS to discover all reachable positions in current environment.

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
    controller.stop()
    return len(all_locs)

def cal_min_dist(index, all_locs, loc2idx):
    '''
    Calculate min distances from one particular position (index) to all other positions in environment 
    * currently support only grid size of 0.5
    '''
    distance = [-1] * len(all_locs)
    distance[index] = 0
    move = [[0, 0.5], [0, -0.5], [0.5, 0], [-0.5, 0]]

    visisted = {}
    for i in range(len(all_locs)):
        visisted[i] = False
    
    queue = []
    queue.append(index)
    
    while len(queue) > 0:
        cur = queue[0]
        queue = queue[1:]
     
        visisted[cur] = True
        cur_loc = all_locs[cur]
        for m in move:
     
            neighbor = (cur_loc[0] + m[0], cur_loc[1] + m[1])
            if neighbor not in loc2idx:
                continue 
     
            nxt = loc2idx[neighbor]
            if not visisted[nxt]:
                distance[nxt] = distance[cur] + 1           
                queue.append(nxt)
                visisted[nxt] = True
     
    return distance


def dump(scene="FloorPlan21", resolution=(300, 300)):
    '''
    Dump needed data to hdf5 file to speed up training. Dumped file can be loaded using: f = h5py.File(filename, 'r'), where:
    - f['locations'][()]: numpy array of all states in format (x, z, rotation, looking angle)
    - f['observations'][()]: numpy array of RGB images of corresponding states in f['locations']
    - f['graph'][()]: numpy array representing transition graph between states. e.g: f[0] = array([ 16., 272.,   4.,  12.,   1.,  -1.], dtype=float32)
        means from 1st locations, the agent will reach 16th state by taking action 0 (move forward), 272th state by taking action 1 (move backward),
        reach 4th state by taking action 2 (rotate right), reach 12th state by taking action 3 (rotate left), 
        reach 1th state by taking action 4 (look down) and cannot take action 5 (look up) indicated by -1 value.
    - f['visible_objects'][()]: visible objects at corresponding states in f['locations']
    - f['shortest'][()]: numpy array with shape of (num_states, num_states) indicating the shortest path length between 
        every pair of states.
    '''

    f = h5py.File("dumped/{}.hdf5".format(scene), "w")

    observations = []
    dump_features = []
    locations = []
    visible_objects = []

    controller = ai2thor.controller.Controller()
    controller.start()
    controller.reset(scene)

    event = controller.step(dict(action='Initialize', gridSize=0.5))
    y_coord = event.metadata['agent']['position']['y']

    locations.append((event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z']))

    # Using BFS to discover all reachable positions in current environment.

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
    print("{} locations".format(len(all_locs)))
    states = []

    # Adding rotations and looking angles
    for loc in all_locs:
        for rot in [0, 90, 180, 270]:
            states.append((loc[0], loc[1], rot))
            # for horot in [-30, 0, 30, 60]:
                # states.append((loc[0], loc[1], rot, horot))

    # ------------------------------------------------------------------------------
    ## Calculate shortest path length array

    sta2idx = dict(zip(states, range(len(states))))
    loc2idx = dict(zip(all_locs, range(len(all_locs))))

    shortest_loc = np.zeros((len(all_locs), len(all_locs)))
    for i in range(len(all_locs)):
        dists = cal_min_dist(i, all_locs, loc2idx)
        for j, d in enumerate(dists):
            if j != i:
                shortest_loc[i, j] = d
                shortest_loc[j, i] = d

    shortest_state = np.zeros((len(states), len(states)))
    for i in range(len(states)):
        for j in range(len(states)):
            if i != j:
                from_loc = loc2idx[states[i][0], states[i][1]]
                to_loc = loc2idx[states[j][0], states[j][1]]
                shortest_state[i, j] = shortest_loc[from_loc, to_loc]
                shortest_state[j, i] = shortest_state[i, j]

    # ------------------------------------------------------------------------------

    # Building transition graph

    graph = np.zeros(shape=(len(states), 4), dtype=int)

    directions = {0: 1, 90: 1, 180: -1, 270: -1}

    for state in states:
        loc = (state[0], state[1])
        rot = state[2]
        
        to_states = []

        if rot == 0 or rot == 180:
            to_states.append((loc[0], loc[1] + directions[rot] * 0.5, rot)) # move ahead
            to_states.append((loc[0], loc[1] - directions[rot] * 0.5, rot)) # move back

        else:
            to_states.append((loc[0] + directions[rot] * 0.5, loc[1], rot)) # move ahead
            to_states.append((loc[0] - directions[rot] * 0.5, loc[1], rot)) # move back

        to_states.append((loc[0], loc[1], rot + 90 if rot <= 180 else 0)) # turn right
        to_states.append((loc[0], loc[1], rot - 90 if rot >= 90 else 270)) # turn left
        
        # to_states.append((loc[0], loc[1], rot + 30)) # look down
        # to_states.append((loc[0], loc[1], rot, horot - 30)) # look up

        state_idx = sta2idx[state]
        for i, new_state in enumerate(to_states):
            if new_state in sta2idx:
                graph[state_idx][i] = sta2idx[new_state]
            else:
                graph[state_idx][i] = -1

    # ------------------------------------------------------------------------------
    laser = {}
    ## Calculate laser 
    for loc in all_locs:
        pos = (loc[0], loc[1], 0)
        north = 0 
        while graph[sta2idx[pos]][0] != -1:
            north += 1
            pos = states[graph[sta2idx[pos]][0]]
            assert pos[2] == 0

        pos = (loc[0], loc[1], 0)
        south = 0 
        while graph[sta2idx[pos]][1] != -1:
            south += 1
            pos = states[graph[sta2idx[pos]][1]]
            assert pos[2] == 0

        pos = (loc[0], loc[1], 90)
        right = 0 
        while graph[sta2idx[pos]][0] != -1:
            right += 1
            pos = states[graph[sta2idx[pos]][0]]
            assert pos[2] == 90

        pos = (loc[0], loc[1], 90)
        left = 0 
        while graph[sta2idx[pos]][1] != -1:
            left += 1
            pos = states[graph[sta2idx[pos]][1]]
            assert pos[2] == 90

        laser[(loc[0], loc[1], 0)] = [north, south, right, left]
        laser[(loc[0], loc[1], 180)] = [south, north, left, right]
        laser[(loc[0], loc[1], 90)] = [right, left, south, north]
        laser[(loc[0], loc[1], 270)] = [left, right, north, south]

    lasers = []
    for state in states:
        lasers.append(laser[state])

    # ------------------------------------------------------------------------------

    # Adding observations 

    for state in states:
        vis_objects = set()
        for horot in [-30, 0, 30, 60]:
            event = controller.step(dict(action='TeleportFull', x=state[0], y=y_coord, z=state[1], rotation=state[2], horizon=horot))
            if horot == 0:
                resized_frame = cv2.resize(event.frame, (resolution[0], resolution[1]))
                observations.append(resized_frame)
        
            visible = [obj for obj in event.metadata['objects'] if obj['visible']]
            for obj in visible:
                vis_objects.add(obj['objectType'])

        if len(vis_objects) > 0:
            visible_objects.append(",".join(list(vis_objects)))
        else:
            visible_objects.append("")

    # ------------------------------------------------------------------------------

    print("{} states".format(len(states)))

    all_visible_objects = list(set(",".join([o for o in visible_objects if o != '']).split(',')))
    all_visible_objects.sort()

    for c in ['Lamp', 'PaperTowelRoll', 'Glassbottle']:
        if c in all_visible_objects:
            all_visible_objects.remove(c)

    target_locations = []
    for target in all_visible_objects:
        target_ids = [idx for idx in range(len(states)) if target in visible_objects[idx].split(",")]
        target_locs = [states[idx] for idx in target_ids]
        target_loc = list(target_locs[np.random.choice(range(len(target_locs)))])
        target_locations.append(target_loc)

    controller.stop()

    f.create_dataset("locations", data=np.asarray(states, np.float32))
    f.create_dataset("observations", data=np.asarray(observations, np.uint8))
    f.create_dataset("graph", data=graph)
    f.create_dataset("visible_objects", data=np.array(visible_objects, dtype=object), dtype=h5py.special_dtype(vlen=str))
    f.create_dataset("all_visible_objects", data=np.array(all_visible_objects, dtype=object), dtype=h5py.special_dtype(vlen=str))
    f.create_dataset("shortest", data=shortest_state)
    f.create_dataset("lasers", data=np.asarray(lasers, np.float32))
    f.create_dataset("target_locations", data=np.asarray(target_locations, np.float32))
    f.close()

    return y_coord

def dump_resnet(tmp, extractor, normalize, scene="FloorPlan28"):
    '''
    Load a hdf5 file and add resnet features and classification scores corresponding to observations.
    '''
    
    f = h5py.File("dumped/{}.hdf5".format(scene), "a")
    observations = f['observations']
    resnet_features = []
    resnet_scores = []

    print("{} states in scene {}".format(observations.shape[0], scene))
    for idx, ob in enumerate(observations):
        print("Dumping resnet: {}/{}".format(idx + 1, observations.shape[0]), end='\r', flush=True)
        resized_frame = np.transpose(ob, (2, 0 ,1))
        resized_frame = torch.from_numpy(resized_frame).type(torch.cuda.FloatTensor)
        resized_frame = normalize(resized_frame)
        inp = resized_frame.unsqueeze(0)
        
        feature = extractor(inp)
        feature = feature.squeeze().detach().cpu().numpy()
        resnet_features.append(feature)

        score = tmp(inp)
        score = score.squeeze().detach().cpu()
        score = F.softmax(score).numpy()
        resnet_scores.append(score)

    f.create_dataset("resnet_features", data=np.asarray(resnet_features, np.float32))
    f.create_dataset("resnet_scores", data=np.asarray(resnet_scores, np.float32))
    f.close()

def dump_feature(scene, y_coord, cat2idx):
    f = h5py.File("dumped/{}.hdf5".format(scene), "a")
    states = f['locations'][()]
    laser = f['lasers'][()]

    dump_features = []
    controller = ai2thor.controller.Controller()
    controller.start()

    controller.reset(scene)
    event = controller.step(dict(action='Initialize', gridSize=0.5, visibilityDistance=1000.0))

    for i, state in enumerate(states):
        event = controller.step(dict(action='TeleportFull', x=state[0], y=y_coord, z=state[1], rotation=state[2], horizon=0))
        visible = [obj for obj in event.metadata['objects'] if obj['visible']]
        df = np.zeros(len(cat2idx) + 4)
        df[-4:] = laser[i].tolist()
        for obj in visible:
            try:
                obj_id = cat2idx[obj['objectType']]
                df[obj_id] = obj['distance']
            except:
                print(obj['objectType'])
                
        dump_features.append(df)

    controller.stop()
    f.create_dataset("dump_features", data=np.asarray(dump_features, np.float32))
    f.close()

if __name__ == '__main__':
    config = json.load(open("config.json"))
    cat2idx = config['new_objects']
    scenes = {'FloorPlan1': 41, 'FloorPlan2': 34, 'FloorPlan3': 1, 'FloorPlan4': 12, 'FloorPlan5': 25, 'FloorPlan6': 7, 'FloorPlan7': 91, 'FloorPlan8': 48, 'FloorPlan9': 19, 'FloorPlan10': 61, 'FloorPlan11': 19, 'FloorPlan12': 23, 'FloorPlan13': 50, 'FloorPlan14': 29, 'FloorPlan15': 18, 'FloorPlan16': 45, 'FloorPlan17': 25, 'FloorPlan18': 62, 'FloorPlan19': 19, 'FloorPlan20': 24, 'FloorPlan21': 14, 'FloorPlan22': 38, 'FloorPlan23': 14, 'FloorPlan24': 20, 'FloorPlan25': 11, 'FloorPlan26': 11, 'FloorPlan27': 6, 'FloorPlan28': 23, 'FloorPlan29': 23, 'FloorPlan30': 21, 'FloorPlan201': 58, 'FloorPlan202': 35, 'FloorPlan203': 150, 'FloorPlan204': 36, 'FloorPlan205': 60, 'FloorPlan206': 23, 'FloorPlan207': 45, 'FloorPlan208': 67, 'FloorPlan209': 100, 'FloorPlan210': 65, 'FloorPlan211': 33, 'FloorPlan212': 26, 'FloorPlan213': 64, 'FloorPlan214': 46, 'FloorPlan215': 108, 'FloorPlan216': 32, 'FloorPlan217': 36, 'FloorPlan218': 42, 'FloorPlan219': 47, 'FloorPlan220': 63, 'FloorPlan221': 28, 'FloorPlan222': 19, 'FloorPlan223': 56, 'FloorPlan224': 81, 'FloorPlan225': 1, 'FloorPlan226': 10, 'FloorPlan227': 61, 'FloorPlan228': 16, 'FloorPlan229': 54, 'FloorPlan230': 108, 'FloorPlan301': 36, 'FloorPlan302': 26, 'FloorPlan303': 20, 'FloorPlan304': 29, 'FloorPlan305': 19, 'FloorPlan306': 22, 'FloorPlan307': 23, 'FloorPlan308': 30, 'FloorPlan309': 67, 'FloorPlan310': 20, 'FloorPlan311': 62, 'FloorPlan312': 20, 'FloorPlan313': 27, 'FloorPlan314': 23, 'FloorPlan315': 24, 'FloorPlan316': 29, 'FloorPlan317': 39, 'FloorPlan318': 22, 'FloorPlan319': 27, 'FloorPlan320': 15, 'FloorPlan321': 37, 'FloorPlan322': 19, 'FloorPlan323': 44, 'FloorPlan324': 51, 'FloorPlan325': 69, 'FloorPlan326': 10, 'FloorPlan327': 28, 'FloorPlan328': 20, 'FloorPlan329': 27, 'FloorPlan330': 64, 'FloorPlan401': 27, 'FloorPlan402': 22, 'FloorPlan403': 21, 'FloorPlan404': 14, 'FloorPlan405': 10, 'FloorPlan406': 27, 'FloorPlan407': 12, 'FloorPlan408': 1, 'FloorPlan409': 10, 'FloorPlan410': 20, 'FloorPlan411': 8, 'FloorPlan412': 7, 'FloorPlan413': 18, 'FloorPlan414': 14, 'FloorPlan415': 14, 'FloorPlan416': 23, 'FloorPlan417': 15, 'FloorPlan418': 18, 'FloorPlan419': 2, 'FloorPlan420': 2, 'FloorPlan421': 8, 'FloorPlan422': 9, 'FloorPlan423': 12, 'FloorPlan424': 6, 'FloorPlan425': 7, 'FloorPlan426': 10, 'FloorPlan427': 12, 'FloorPlan428': 17, 'FloorPlan429': 13, 'FloorPlan430': 28}
    
    # scene_size = {}
    # for room in config['rooms'].keys():
    #     # train_objects = config['rooms'][room]['train_objects']
    #     # test_objects = config['rooms'][room]['test_objects']
    #     for scene in config['rooms'][room]['scenes']:
    #         scene_size[scene] = check_size(scene)

    # print(scene_size)



    tmp = models.resnet50(pretrained=True)
    tmp.eval()
    tmp.cuda()
    # tmp = models.inception_v3(pretrained=True)
    modules = list(tmp.children())[:-1]
    extractor = nn.Sequential(*modules)
    extractor.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    for scene in [   'FloorPlan1',
                     "FloorPlan2",
                     "FloorPlan10",
                     "FloorPlan28",
                     "FloorPlan201"]:
                     # "FloorPlan202",
                     # "FloorPlan204",
                     # "FloorPlan206",
                     # "FloorPlan301",
                     # "FloorPlan302",
                     # "FloorPlan309",
                     # "FloorPlan311",
                     # "FloorPlan401",
                     # "FloorPlan402",
                     # "FloorPlan406",
                     # "FloorPlan430"][:1]:

        y_coord = dump(scene, config['resolution'])
        dump_feature(scene, y_coord, cat2idx)
        dump_resnet(tmp, extractor, normalize, scene)