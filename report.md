# DQN Navigation Agent

## Architecture and Parameters

This implementation is based on the Nature [paper](https://www.nature.com/articles/nature14236). The model consists of three fully connected layers and ReLU functions. The input layer has 37 units which represent the size of the state space. The output layer has 4 units corresponding to the Q value of each action. A drop out of `0.25` showed significant performance inprovement.

The following parameters are used: 
- `BUFFER_SIZE = int(1e5)`:  Replay buffer size (used for experience replay).
- `BATCH_SIZE = 64`: Subset of previous agent experiences chosen randomly during the learning steps.
- `GAMMA = 0.99`: Discount factor.
- `TAU = 1e-3`: For soft update of target parameters.
- `LR = 5e-4` : Learning rate. 
- `UPDATE_EVERY = 4`: To avoid very correlated experience samples.
- `n_episodes <= 1800`: A successful implementation should reach a score of 13 in less than 1800 episodes.
- `max_t (int) = 1000`: Maximum number of timesteps per episode.
- `eps_start (float) = 1.0`: Starting value of epsilon, for epsilon-greedy action selection.
- `eps_end (float) = 0.01`: Minimum value of epsilon.
- `eps_decay (float) = 0.995`: Multiplicative factor (per episode) for decreasing epsilon. The decay rule follows the formula `eps = max(eps_end, eps_decay*eps)`.


### Untrained (Random) agent
This is the performance of an untrained agent. At this stage, the actions are basically chosen randomly.
![untrained](gifs/random_navigation.gif)

### Trained (Smart) agent
This is an agent performing after the training process.
![trained](gifs/smart_navigation.gif)

## Training Results
The figure below illustrates the improvement in score as the number of episodes increase which shows the learning behavior.

![results](/Score_vs_Episodes.png)

```
Episode 100	Average Score: 0.81
Episode 200	Average Score: 4.80
Episode 300	Average Score: 8.46
Episode 400	Average Score: 11.19
Episode 500	Average Score: 11.96
Episode 552	Average Score: 13.05
Environment solved in 452 episodes!	Average Score: 13.05
```

## Future Work

The following techniques and algorithms may improve the performance of the current implementation.
- [RAINBOW](https://arxiv.org/pdf/1710.02298.pdf).
- [DDQN] (https://arxiv.org/abs/1509.06461).
- Hyperparameter optimization.
