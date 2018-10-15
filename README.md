# ARS-BipedalWalker

Augmented Random Search on the BipedalWalker-v2 gym environment.

Change the environment in `main.py` to use other environments. However, for some environments you may need to
modify the code a little bit.

## Training the agent
Run `main.py` to train the agent. After training, it saves the normalizer class object and the agent's weights so that it can be loaded
later.

To disable logging or monitoring while training, set the `log_every` or `record_every` variables to `None` respectively.

## Loading the agent
Run `load.py` to load the agent's weights and the normalizer class. It also renders a test episode to ensure everything loaded fine.
To train the agent after loading, just run the `agent.train(...)` function.
