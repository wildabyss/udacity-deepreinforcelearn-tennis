## Project 3: Collaboration and Competition Report

### Algorithm

The project uses deep reinforcement learning to train an agent that solves the Unity Tennis game. It leverages the Deep Deterministic Policy Gradient (DDPG) method to solve the continuous-action space game. The method consists of a Critic, which learns the optimal action-value function of the game, and an Actor, which selects the action corresponding to the global maximum of the action-value corresponding to the current state. DDPG is an extension of DQN in that, rather than using argmax to select the discrete optimal action from the optimal action-value function, it uses a learning-based optimizer.

Each Actor and Critic consists of a local and target feedforward neural network, whereby the local networks are used to generate the immediate action and action-value, and the target networks are used in training to generate the target values in TD error. The local weights are updated per training step, and the target weights linearly blend in the local weights through a hyperparameter SOFT_UPDATE_RATE.

The state space and action space from each player's perspective is mirrored, therefore a single Actor, Critic and replay buffer is used to provide action for both players. The agent engages in self play to train.

A few improvements were made on top of the vanilla DDPG to incentivize and stabilize the training:

1. The Ornstein-Uhlenbeck (OU) noise is added to the actions during training to allow exploration. I used a standard normal noise in the OU process model so that even large noise actions have a non-zero chance of getting picked, however small the standard deviation is. The annealing process during training involves decreasing the random walk magnitude.

2. Different from the previous project, I elected to retain a separate memory for bad experiences, which correspond to those experience tuples that result in a negative reward. Different from the previous project, in which the result of an action from goal state is immediate, a good shot from the tennis game does not yield a positive reward until some time in the future; therefore, it is futile to try to retain a separate good memory.

### Hyperparameters

The following hyper-parameters have been shown to work well during training.

REPLAY_BUFFER_SIZE = int(1e6)   # replay buffer size
REPLAY_BATCH_SIZE = int(50)     # minibatch size
FUTURE_DISCOUNT = 0.99          # discount factor
SOFT_UPDATE_RATE = 1e-3         # soft update rate
LR_ACTOR = 1e-4                 # learning rate of the actor 
LR_CRITIC = 3e-4                # learning rate of the critic
WEIGHT_DECAY = 0.0              # L2 weight decay
UPDATE_FREQ = 1                 # Learn frequency in iterations
UPDATE_EVERY = True             # If true, learn once for every agent

USE_TWO_MEMS = True             # Whether to separately sample good vs bad experiences
BAD_MEM_RATIO = 0.2
UNIFORM_SAMPLE_MIN_SCORE = 50   # Minimum score above which we revert to sampling good and bad experiences uniformly

ADD_NOISE = True                # Whether to add OU noise
NOISE_SIGMA_START = 4           # Start of the OU noise random walk magnitude
NOISE_SIGMA_DECAY = 0.9999      # Decay rate of the OU random walk
NOISE_SIGMA_MIN = 0.01          # Minimum of the OU random walk

N_EPISODES = 1250
VICTORY_SCORE = 0.5

The following points were instrumental in getting the training to succeed:

1. Early exploration is extremely important to get a statistical representation of the value function. If the noise decays too fast, it results in poor learning. NOISE_SIGMA_START is set to 4 to encourage breaking out of early local minima. NOISE_SIGMA_DECAY is set to 0.9999. Instead of decaying per episode, I elected to decay the noise per step, since a long play with many steps is a sign that the learn is succeeding and the noise should decay accordingly.
2. Different from the previous project where the update frequency had to be low in order to stabilize the learn, I found that having a very fast update frequency can much more readily stabilize this project. I elected to train per agent action as well, which results in 2 training steps per environment step.
3. The REPLAY_BATCH_SIZE must not be too big.
4. I had to set WEIGHT_DECAY to 0 in order for training scores to increase to meaningful values. It's possible 1e-4 is too big and a smaller value would work better.

Using two separate memories only yielded marginally faster learn than using a uniform sampling.

### Results

![Trained Agent](plots/scores.png)

### Improvements

1. Since an action may not show benefit until many states down the road, it may benefit from storing a horizon of sequences and retrieving the batch in the fixed order.

2. We can attempt to adjust the reward system by also penalizing the number of actions to minimize jitter.