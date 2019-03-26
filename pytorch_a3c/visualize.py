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
parser.add_argument('--save', type=int, default=0)

smooth = 5

def old_foo_all(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.pkl')]

    rewards = [] 
    sc_rates = []
    for f in files:    
        sc = pickle.load(open(folder+'/' + f, 'rb'))
        rewards.append(sc)        

    min_length = min([len(s) for s in rewards])

    labels = ['rewards', 'success_rate']

    # for i, tasks in enumerate([rewards, sc_rates]):
    for i, tasks in enumerate([rewards]):
        avg = np.mean([s[:min_length] for s in tasks], 0)[::100]
        smoothed_y = [np.mean(avg[max(0, yi - smooth):min(yi + smooth, len(avg)-1)]) for yi in range(len(avg))]
        plt.plot(range(len(smoothed_y)), smoothed_y, c='C' + str(i), label=labels[i])
        plt.plot(range(len(avg)), avg, alpha=0.3, c='C' + str(i), label=labels[i])

    plt.legend()
    plt.show()

def foo_all(folder, save):
    fig = plt.figure()
    files = [f for f in os.listdir(folder) if f.endswith('.pkl')]

    rewards = [] 
    sc_rates = []
    redundancies = []
    entropies = []
    for f in files:    
        sc = pickle.load(open(folder+'/' + f, 'rb'))
        rewards.append(sc['rewards'])
        sc_rates.append(sc['success_rate'])
        try:
            redundancies.append(sc['redundancies'])
            entropies.append(sc['entropies'])
        except:
            pass
        # print(len(sc['rewards']), len(sc['success_rate']))

    all_labels = [['rewards', 'success_rate (scale x 10)', 'entropies (scale x 10)'], ['redundancies']]
    all_tasks = [[rewards, sc_rates, entropies], [redundancies]]
    for li, (labels, all_tasks) in enumerate(zip(all_labels, all_tasks)):
        plt.subplot(1, 2, li+1)
        for i, tasks in enumerate(all_tasks):
        # for i, tasks in enumerate([sc_rates]):
            try:
                min_length = min([len(s) for s in tasks]) // 5
            except:
                continue
            avg = np.mean([s[:min_length] for s in tasks], 0)[::20]
            if li == 0 and i > 0:
                avg *= 10
            smoothed_y = [np.mean(avg[max(0, yi - smooth):min(yi + smooth, len(avg)-1)]) for yi in range(len(avg))]
            plt.plot(range(len(smoothed_y)), smoothed_y, c='C' + str(i), label=labels[i])
            plt.plot(range(len(avg)), avg, alpha=0.3, c='C' + str(i))

        plt.legend()

    if save:
        title = input("Figure title:")
        fig.set_size_inches(10, 5)
        plt.savefig('../images/{}.png'.format(title), bbox_inches='tight')
    else:
        plt.show()

def foo(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    
    tasks = {} 
    for f in files:
        t = '_'.join(f.split('_')[:2])
        if t not in tasks:
            tasks[t] = [pickle.load(open(folder+'/' + f, 'rb'))['rewards']]
        else:
            tasks[t].append(pickle.load(open(folder+'/' + f, 'rb'))['rewards'])

    for k, v in tasks.items():
        min_length = min([len(vi) for vi in v])
        avg = np.mean([vi[:min_length] for vi in v], 0)[::200]
        smoothed_y = [np.mean(avg[max(0, yi - smooth):min(yi + smooth, len(avg)-1)]) for yi in range(len(avg))]
        plt.plot(range(len(avg)), avg, label=k)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 0:
        try:
            foo_all(args.folder, args.save)
        except:
            old_foo_all(args.folder)
    else:
        foo(args.folder)