import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def process(dpath):
	folders = [f for f in os.listdir(dpath)]
	# folders = ['Z_16']
	summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in folders]

	for f, summary in zip(folders, summary_iterators):
		tag_dict = {}
		tags = summary.Tags()['scalars']
		for tag in tags:
			log_type = "_".join(tag.split('/')[1:])
			if log_type not in tag_dict:
				tag_dict[log_type] = {'steps' : [], 'values': [] }
				
			steps = [e.step for e in summary.Scalars(tag)]
			values = [e.value for e in summary.Scalars(tag)]

			tag_dict[log_type]['steps'].append(steps)
			tag_dict[log_type]['values'].append(values)

		# print(list(tag_dict.keys()))
		# break
		for k, v in tag_dict.items():
			df = pd.DataFrame(columns=['Step', 'Value'])
			# print(v['steps'], v['values'])
			# sys.exit()
			df['Step'] = np.mean(np.vstack(v['steps']), 0)
			df['Value'] = np.mean(np.vstack(v['values']), 0)
			df.to_csv(os.path.join(dpath, "{}.csv".format(k)))

if __name__ == '__main__':
	path = "/home/yoshi/thesis/RL-target-driven-navigation-ai2thor/tf_a2c/training-history/FloorPlan2_6"
	process(path)