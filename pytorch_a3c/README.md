# File Descriptions

- `main.py`: file to run, please read the arguments and corresponding descriptions
- `train.py`: training file, initialized in main thread as a independent process (A3C), support multi-gpus
- `test.py`: file to test trained model
- `dumping.py`: file to run ai2thor controller and dump needed information to hdf5 files (note that the number of actions is 4 and angle is 22.5 by default)
- `layers.py`: contains GCN code
- `utils.py`: contains necessary functions
- `keyboard_agent.py`: to run and interact with dumped file 
- `config.json`: configurations including rewarding scheme, file paths, data split
- `env/ai2thor_env.py`: environment
