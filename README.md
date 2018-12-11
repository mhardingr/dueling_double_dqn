# dueling_double_dqn

This repository contains a Python script (tested only with 2.7) for solving the CartPole-v0 and MountainCar-v0 openai gym
environments using deep q-learning algorithms: vanilla DQN ([V. Mnih, et al.](https://arxiv.org/abs/1312.5602)), Double DQN ([H. van Hasselt, et al.](https://arxiv.org/abs/1509.06461)), and Dueling DQN ([Z. Wang, et al.](https://arxiv.org/abs/1511.06581)).

The hyperparameters for each model used to learn each environment are found in the `all_dqns.py` file.

```
usage: all_dqns.py [-h] [--env ENV] [--model MODEL_NAME]

Deep Q Network Argument Parser

optional arguments:
  -h, --help          show this help message and exit
  --env ENV			("CartPole-v0" for CP [default], "MountainCar-v0" for MC)
  --model MODEL_NAME ("dqn" for DQN [default], "ddqn" for DDQN, "dueling" for Dueling)
```

