import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pickle

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--mode', type=int, default=0,
                    help='visualization mode: \
                    	0: all \
                    	1: separated tasks')
parser.add_argument('--folder', type=str, default='training-history/multitask_onehot/')

smooth = 10
def foo_all(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.pkl')]

    tasks = [] 
    for f in files:    
        sc = pickle.load(open(folder+'/' + f, 'rb'))
        print(f, len(sc))     
        tasks.append(sc[:14000])
      
    avg = np.mean(tasks, 0)
    smoothed_y = [np.mean(avg[max(0, yi - smooth):min(yi + smooth, len(avg)-1)]) for yi in range(len(avg))]
    plt.plot(range(len(smoothed_y)), smoothed_y, c='C1')
    plt.plot(range(len(avg)), avg, alpha=0.3, c='C1')
    plt.show()

def foo(folder):
    files = [f for f in os.listdir(folder) if '.pth' not in f]
    
    tasks = {} 
    for f in files:
        t = '_'.join(f.split('_')[:2])
        if t not in tasks:
            tasks[t] = [pickle.load(open(folder+'/' + f, 'rb'))]
        else:
            tasks[t].append(pickle.load(open(folder+'/' + f, 'rb')))

    for k, v in tasks.items():
        avg = np.mean(v, 0).tolist()  
        plt.plot(range(len(avg)), avg, label=k)
    plt.legend()
    plt.show()


if __name__ == '__main__':
	args = parser.parse_args()
	if args.mode == 0:
		foo_all(args.folder)
	else:
		foo(args.folder)