import os
import json
import h5py
import operator

config = json.load(open('config.json'))

scene_type = ['Kitchens', 'Living Rooms', 'Bedrooms', 'Bathrooms']
visible = []
cnt = {}
st = 0

for s in ['train_scenes', 'test_scenes']:
    cnt[s] = {}
    for f in config['rooms'][scene_type[st]][s]:
        f = h5py.File("dumped/{}.hdf5".format(f), 'r')
        visible.append(f['all_visible_objects'][()].tolist())
        for o in f['all_visible_objects'][()].tolist():
            if o not in cnt[s]:
                cnt[s][o] = 1
            else:
                cnt[s][o] +=1
    if s == 'train_scenes':
        cnt[s] = [o for o, c in cnt[s].items() if c > 7]
    else:
        cnt[s] = [o for o, c in cnt[s].items()]

print(cnt)

print("Joint: ", set(cnt['train_scenes']).intersection(set(cnt['test_scenes'])))