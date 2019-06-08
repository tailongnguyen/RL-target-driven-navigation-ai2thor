import json
import pickle
import numpy as np
import progressbar

from collections import Counter

def build_graph():
    data = json.load(open("relationships.json", 'rb'))
    mapping = pickle.load(open("new_mapping.pkl", "rb"))

    vg2idx = mapping['vg2idx']
    idx2obj = mapping['idx2obj']
    obj2idx = mapping['obj2idx']
    # rela2idx = mapping['rela2idx']

    cooc = {}
    cooc_pred = {}
    for i in range(105):
        for j in range(105):
            cooc[i, j] = []
            cooc_pred[i, j] = []

    bar = progressbar.ProgressBar()
    # invalid_predicates = []
    for i in bar(range(len(data))):
        d = data[i]
        for r in d['relationships']:
            if "name" in r['object']:
                k = "name"
            else:
                k = "names"
                
            if type(r['object'][k]) == list:
                obj = r['object'][k][0]
            else:
                obj = r['object'][k]
            
            
            if "name" in r['subject']:
                k = 'name'
            else:
                k = "names"
                
            if type(r['subject'][k]) == list:
                sub = r['subject'][k][0]
            else:
                sub = r['subject'][k]

            try:
                objs = vg2idx[obj]
                subs = vg2idx[sub]
            except:
                continue

            for o in objs:
                for s in subs:
                    try:
                        obj_id = obj2idx[o]
                        sub_id = obj2idx[s]
                    except:
                        continue
                    # try:
                        # cooc_pred[obj_id, sub_id].extend(rela2idx[r['predicate'].lower()])
                    # except:
                        # invalid_predicates.append(r['predicate'].lower())
                    if type(r['predicate']) == list:
                        cooc_pred[obj_id, sub_id].extend([p.lower() for p in r['predicate']])
                    else:   
                        cooc_pred[obj_id, sub_id].append(r['predicate'].lower())
                        
                    # cooc[obj_id, sub_id].append(r['relationship_id'])


    relations = np.identity(105, np.float32)
    # raw_relations = np.identity(87, np.float32)
    for k, v in cooc_pred.items():
      if len(v) > 0:
          cnt_v = Counter(v + cooc_pred[k[1], k[0]])
          freqs = np.array(list(cnt_v.values()))
          if np.sum(freqs > 3) > 0:
              relations[k[0], k[1]] = 1
              relations[k[1], k[0]] = 1

    # for k, v in cooc.items():
    #   if k[0] != k[1]:
    #       raw_relations[k[0], k[1]] = len(v + cooc[k[1], k[0]])
    #       raw_relations[k[1], k[0]] = len(v + cooc[k[1], k[0]])

    with open("new_cooc_pred.pkl", 'wb') as f:
        pickle.dump(cooc_pred, f, pickle.HIGHEST_PROTOCOL)

    # with open("invalid.txt", 'wb') as f:
    #   pickle.dump(invalid_predicates, f, pickle.HIGHEST_PROTOCOL)

    np.save("new_relations", relations)
    # np.save("raw_relations", raw_relations)

def lcs(X, Y, m, n): 
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)] 
      
    # To store the length of  
    # longest common substring 
    result = 0 
  
    # Following steps to build 
    # LCSuff[m+1][n+1] in bottom up fashion 
    for i in range(m + 1): 
        for j in range(n + 1): 
            if (i == 0 or j == 0): 
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]): 
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j]) 
            else: 
                LCSuff[i][j] = 0
    return result 

def mapping_predicates():
    mapping = pickle.load(open("mapping.pkl", "rb"))

    vg2idx = mapping['vg2idx']
    idx2obj = mapping['idx2obj']
    rela2idx = mapping['rela2idx']

    new_rela = {}
    for k, v in rela2idx.items():
        new_rela[k] = [v]

    known_pred = list(rela2idx.keys())
    not_found = 0
    invalid_predicates = pickle.load(open('invalid.txt', 'rb'))

    bar = progressbar.ProgressBar()
    for i in bar(range(len(invalid_predicates))):
        p = invalid_predicates[i]
        new_rela[p] = []
        found = 0
        for kp in known_pred:
            if lcs(p, kp, len(p), len(kp)) / max(len(p), len(kp)) > 0.6:
                new_rela[p].append(rela2idx[kp])
                found = 1
        if found == 0:
            not_found += 1

    mapping['all_rela2idx'] = new_rela

    print("{} not found.".format(not_found))

    with open("mapping.pkl", 'wb') as f:
        pickle.dump(mapping, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    build_graph()
    # mapping_predicates()