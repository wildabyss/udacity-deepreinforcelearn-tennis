## Project 2: Continuous Control Report

### Algorithm

The project uses deep reinforcement learning to train an agent that solves the Unity Reacher game. It leverages the Deep Deterministic Policy Gradient (DDPG) method to solve the continuous-action space game. The method consists of a Critic, which learns the optimal action-value function of the game, and an Actor, which selects the action corresponding to the global maximum of the action-value corresponding to the current state. DDPG is an extension of DQN in that, rather than using argmax to select the discrete optimal action from the optimal action-value function, it uses a learning-based optimizer.

Each Actor and Critic consists of a local and target feedforward neural network, whereby the local networks are used to generate the immediate action and action-value, and the target networks are used in training to generate the target values in TD error. The local weights are updated per training step, and the target weights linearly blend in the local weights through a hyperparameter SOFT_UPDATE_RATE.

A few improvements were made on top of the vanilla DDPG to incentivize and stabilize the training:

1. The Ornstein-Uhlenbeck (OU) noise is added to the actions during training to allow exploration. I used a Gaussian noise in the OU process model so that even large noise actions have a non-zero chance of getting picked, however small the standard deviation is. The annealing process during training simply involves decreasing the standard deviation (sigma) toward the desired minimum value.

2. At start of training, the vast majority of the experience tuples will be meaningless jitters in the robotic arms, and only a very small number would be ones corresponding to posive rewards. As a result, it is likely there will not be a single positive reward tuple in a given randomly selected batch. Experimentation has shown that needs leads to very slow learn where the scores do not meaningfully improve. I modified the replay buffer to have two separate memories, one corresponding to bad experiences where no positive reward was obtained, and the other corresponding to good experiences where positive reward was obtained. In each learning step, at least GOOD_MEM_RATIO of the sampled experience tuples must be from good experiences.

3. To avoid score crashing half way through training, the training step is only ran once every two time steps.

### Hyperparameters

The following hyper-parameters have been shown to work well during training.

REPLAY_BUFFER_SIZE = int(1e6)   # replay buffer size
REPLAY_BATCH_SIZE = 120         # minibatch size
FUTURE_DISCOUNT = 0.99          # discount factor
SOFT_UPDATE_RATE = 1e-3         # soft update rate
LR_ACTOR = 1e-4                 # learning rate of the actor 
LR_CRITIC = 2e-4                # learning rate of the critic
WEIGHT_DECAY = 0.0001           # L2 weight decay

USE_TWO_MEMS = True             # Whether to separately sample good vs bad experiences
GOOD_MEM_RATIO = 0.25           # Good vs bad experience ratio in sampling
UNIFORM_SAMPLE_MIN_SCORE = 50   # Minimum score above which we revert to sampling good and bad experiences uniformly

ADD_NOISE = True                # Whether to add OU noise
NOISE_SIGMA_START = 0.15        # Start of the OU noise standard deviation
NOISE_SIGMA_DECAY = 0.98        # Decay rate of the OU standard deviation
NOISE_SIGMA_MIN = 0.001         # Minimum of the OU noise standard deviation

N_EPISODES = 1000
MAX_T = 1000
VICTORY_SCORE = 30

### Results

Training result

![Trained Agent](plots/scores.png)

Training result without the two separate memories. Scores do not meaningfully improve.

![Trained Agent](plots/ScoresSingleMem.png)


### Improvements

1. Since an action may not show benefit until many states down the road, it may benefit from storing a horizon of sequences and retrieving the batch in the fixed order.

2. The current reward system only rewards and penalizes when the tip is near the goal location. We can attempt to adjust the reward system by also penalizing the number of actions to minimize jitter.