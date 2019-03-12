import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def foo_all(folder, smooth=5):
	chosen_cols = ['rewards', 'success_rate']
	fig, axes = plt.subplots(nrows= 1, ncols=len(chosen_cols))
	lines = []
	labels = []
	for ax, t in zip(axes, chosen_cols):
		ax.set_title(t if t != 'redundants' else 'redundant steps')
		files = [os.path.join(folder, f) for f in os.listdir(folder) if t in f]
		files = sorted(files, key=lambda x: x.split('/')[-1].split('_')[0])
		for i, f in enumerate(files):
			labels.append(f.split("/")[-1].split('_')[0])
			log = pd.read_csv(f)

			avg = log['Value'].tolist()
			smoothed_y = [np.mean(avg[max(0, yi - smooth):min(yi + smooth, len(avg)-1)]) for yi in range(len(avg))]

			ax.xaxis.set_major_locator(plt.MaxNLocator(4))
			ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x / 1e6, ',') + ' M'))

			line = ax.plot(log['Step'].tolist(), smoothed_y, c='C{}'.format(i))
			ax.plot(log['Step'].tolist(), avg, c='C{}'.format(i), alpha=0.3)
			lines.append(line[0])

	fig.set_size_inches(12, 4)
	
	# leg = fig.legend(lines[::2], ["{} {}".format(*folder.split('/')[-1].split('_')) for folder in folders], loc = 8, ncol = 2, bbox_to_anchor = (0.50, -0.00), fontsize ='large')
	leg = fig.legend(lines[:len(labels)//2], labels[:len(labels)//2], loc = 8, ncol = 3, bbox_to_anchor = (0.4, -0.00), fontsize ='large')

	# set the linewidth of each legend object
	for legobj in leg.get_lines():
	    legobj.set_linewidth(4.0)

	plt.subplots_adjust(wspace = 0.1, hspace = 0.3, bottom = 0.3)
	plt.savefig("All " + folder.split('/')[-1] + '.png', bbox_inches='tight', dpi = 250)

def foo(folders, smooth=5):
	chosen_cols = ['rewards', 'success_rate']
	fig, axes = plt.subplots(nrows= 1, ncols=len(chosen_cols))
	colors = ['C0', 'C1']
	lines = []
	for i, folder in enumerate(folders):
		for ax, t in zip(axes, chosen_cols):
			ax.set_title(t if t != 'redundants' else 'redundant steps')
			files = [os.path.join(folder, f) for f in os.listdir(folder) if t in f]
			logs = []
			for f in files:
				logs.append(pd.read_csv(f))

			min_size = min([l.shape[0] for l in logs])

			new_logs = []
			for l in logs:
				new_logs.append(l['Value'].tolist()[:min_size])

			avg = np.mean(np.vstack(new_logs), 0).tolist()
			smoothed_y = [np.mean(avg[max(0, yi - smooth):min(yi + smooth, len(avg)-1)]) for yi in range(len(avg))]

			ax.xaxis.set_major_locator(plt.MaxNLocator(4))
			ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(x / 1e6, ',') + ' M'))

			line = ax.plot(logs[0]['Step'].tolist(), smoothed_y, c=colors[i])
			ax.plot(logs[0]['Step'].tolist(), avg, c=colors[i], alpha=0.3)
			lines.append(line[0])

	fig.set_size_inches(12, 4)
	
	leg = fig.legend(lines[::2], ["{} {}".format(*folder.split('/')[-1].split('_')) for folder in folders], loc = 8, ncol = 2, bbox_to_anchor = (0.42, -0.00), fontsize ='large')
	# leg = fig.legend(lines[::2], ["4-stacked-frames", '1-frame'], loc = 8, ncol = 2, bbox_to_anchor = (0.40, -0.00), fontsize ='large')

	# set the linewidth of each legend object
	for legobj in leg.get_lines():
	    legobj.set_linewidth(4.0)

	plt.subplots_adjust(wspace = 0.1, hspace = 0.3, bottom = 0.2)
	plt.savefig("Compare " + folder.split('/')[-1].split('_')[0] + '.png', bbox_inches='tight', dpi = 250)

if __name__ == '__main__':
	foo(["/home/yoshi/thesis/RL-target-driven-navigation-ai2thor/tf_a2c/training-history/FloorPlan1_4", "/home/yoshi/thesis/RL-target-driven-navigation-ai2thor/tf_a2c/training-history/FloorPlan1_6"])
	# foo_all("/home/yoshi/thesis/RL-target-driven-navigation-ai2thor/tf_a2c/training-history/FloorPlan28_6")