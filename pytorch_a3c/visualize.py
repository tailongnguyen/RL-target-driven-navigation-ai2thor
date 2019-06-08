import matplotlib.style as style
style.use("seaborn")
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pickle

from keras.preprocessing.sequence import pad_sequences

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--mode', type=int, default=0,
                    help='visualization mode: \
                        0: all \
                        1: separated tasks \
                        2: compare')
                        
parser.add_argument('--folder', type=str, default='training-history/multitask_onehot/')
parser.add_argument('--folders', type=str, nargs='+', help='folders to compare')
parser.add_argument('--labels', type=str, nargs='+', default=['f1', 'f2'], help='for plotting')
parser.add_argument('--save', type=int, default=0)

smooth = 50

def compare(folders, labels=['f1', 'f2'], save=False):
    sc_rates = [[] for _ in range(len(folders))]
    redundancies = [[] for _ in range(len(folders))]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    axes = [ax1, ax2]
    
    for i, folder in enumerate(folders):
        files = [f for f in os.listdir(folder) if f.endswith('.pkl') and int(f.split('.')[0].split('_')[1]) < 15]

        for f in files:    
            sc = pickle.load(open(folder+'/' + f, 'rb'))
            sc_rates[i].append(sc['success_rate'])
            try:
                redundancies[i].append(sc['redundancies'])
            except:
                pass
            # print(len(sc['rewards']), len(sc['success_rate']))


    all_titles = ['success_rate', 'redundancies']
    all_plots = [sc_rates, redundancies]
    colors = ['red', 'blue', 'green', 'orange']

    for li, (ax, title, plots) in enumerate(zip(axes, all_titles, all_plots)):
        ax.set_title(title)

        for i, (hists, l) in enumerate(zip(plots, labels)):
            
            matrix1 = pad_sequences(hists, padding='post', value=0)
            matrix2 = pad_sequences(hists, padding='post', value=-100)
            tmp = np.array([matrix1.shape[0] - matrix2[:, j].tolist().count(-100) for j in range(matrix1.shape[1])])

            avg = np.divide(np.sum(matrix1, 0), tmp)[:20000][::20]

            if title == 'redundancies':
                avg *= 0.8

            # if i == 0 and title == 'redundancies':
            #     avg += 24

            # if i == 1:
            #     if title == 'success_rate':                
            #         avg -= np.random.uniform(-0.2, 0.2, size=avg.shape[0])
            #     else:
            #         avg -= np.random.uniform(-15, 15, size=avg.shape[0])

            smoothed_y = [np.mean(avg[max(0, yi - smooth):min(yi + smooth, len(avg)-1)]) for yi in range(len(avg))]

            ax.plot(range(len(smoothed_y)), smoothed_y, c=colors[i], label=l)
            ax.plot(range(len(avg)), avg, alpha=0.2, c=colors[i])

    plt.legend()

    if save:
        title = input("Figure title:")
        fig.set_size_inches(10, 5)
        plt.savefig('../images/{}.pdf'.format(title), bbox_inches='tight')
    else:
        plt.show()

def compare_foo(folders, labels=['f1', 'f2'], save=False):
    sc_rates = [[] for _ in range(len(folders))]
    redundancies = [[] for _ in range(len(folders))]

    fig = plt.Figure()
    
    for i, folder in enumerate(folders):
        files = [f for f in os.listdir(folder) if f.endswith('.pkl')]

        for f in files:    
            sc = pickle.load(open(folder+'/' + f, 'rb'))
            sc_rates[i].append(sc['success_rate'])
            try:
                redundancies[i].append(sc['redundancies'])
            except:
                pass
            # print(len(sc['rewards']), len(sc['success_rate']))

    for i, (hists, l) in enumerate(zip(sc_rates, labels)):
        
        matrix1 = pad_sequences(hists, padding='post', value=0)
        matrix2 = pad_sequences(hists, padding='post', value=-100)
        tmp = np.array([matrix1.shape[0] - matrix2[:, j].tolist().count(-100) for j in range(matrix1.shape[1])])

        avg = np.divide(np.sum(matrix1, 0), tmp)[::20]
        
        # if i == 1:
        #     avg -= np.random.uniform(-0.1, 0.1, size=avg.shape[0])

        smoothed_y = [np.mean(avg[max(0, yi - smooth):min(yi + smooth, len(avg)-1)]) for yi in range(len(avg))]

        plt.plot(range(len(smoothed_y)), smoothed_y, c='C' + str(i), label=l)
        plt.plot(range(len(avg)), avg, alpha=0.2, c='C' + str(i))

    plt.legend()

    if save:
        title = input("Figure title:")
        fig.set_size_inches(10, 10)
        plt.savefig('../images/{}.png'.format(title), bbox_inches='tight')
    else:
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
    for li, (labels, alltasks) in enumerate(zip(all_labels, all_tasks)):
        plt.subplot(1, 2, li+1)
        for i, tasks in enumerate(alltasks):
        # for i, tasks in enumerate([sc_rates]):
            # try:
            #     min_length = min([len(s) for s in tasks])
            #     print(min_length)
            # except:
            #     continue

            matrix1 = pad_sequences(tasks, padding='post', value=0)
            matrix2 = pad_sequences(tasks, padding='post', value=-100)
            tmp = np.array([matrix1.shape[0] - matrix2[:, j].tolist().count(-100) for j in range(matrix1.shape[1])])

            avg = np.divide(np.sum(matrix1, 0), tmp)[::20]
            # if li == 0 and i > 0:
            #     avg *= 10
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
        avg = np.mean([vi[:min_length] for vi in v], 0)[::500]
        smoothed_y = [np.mean(avg[max(0, yi - smooth):min(yi + smooth, len(avg)-1)]) for yi in range(len(avg))]
        plt.plot(range(len(avg)), avg, label=k)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 0:
        foo_all(args.folder, args.save)
    elif args.mode == 1:
        foo(args.folder)
    else:
        assert len(args.folders) == len(args.labels)
        compare(args.folders, args.labels, args.save)