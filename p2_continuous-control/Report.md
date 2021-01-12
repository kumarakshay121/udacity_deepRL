# Project 2 Report (Reacher)

## Introduction

The objective of this project is to train an agent to move a double-jointed arm to target locations (Unity's [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment).

## Method

To solve the problem, a vanilla Deep Deterministic Policy Gradient (DDPG) algorithm was chosen, which was introduced by DeepMind in the paper [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf). DDPG uses an actor-critic framework and extends well to continuous actions spaces, unlike DQN. The actor uses a deep neural network as a function approximator to estimate a deterministic policy, i.e., it aims to estimate the best possible action for any given state. The critic uses a deep neural network as a function approximator to estimate a Q-value function and is trained using the actors best believed action as target.

Characteristic techniques of DDPG that help train an agent better:
- **Experience replay**: A replay buffer is used to store previous experiences in order to sample and reuse them for training. Random sampling from a replay buffer breaks any harmful correlations between consecutive experiences and prevents action values from oscillating or diverging catastrophically. Moreover, the replay buffer is a more efficient use of past experiences as it provides the possibility to use experiences multiple times and recall rare occurences.
- **Soft updates**: A soft update involves slowly blending the regular weights with the target network weights instead of copying them at once. This approach leads to faster convergence.

**Network Architecture**

- Actor
```bash
	hidden layer 1: 256 units + RELU activation
	hidden layer 2: 128 units + RELU activation
	output layer: 4 units (action size)
```
- Critic
```bash
	hidden layer 1: 256 units + Leaky RELU activation
	hidden layer 2: 128 units + Leaky RELU activation
	hidden layer 3: 64 units + Leaky RELU activation
	output layer: 1 units (Q value)
```

**Hyperparameters**
```bash
	BUFFER_SIZE = int(1e6)  # replay buffer size
	BATCH_SIZE = 128        # minibatch size
	GAMMA = 0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR_ACTOR = 1e-4         # learning rate of the actor
	LR_CRITIC = 1e-4        # learning rate of the critic
	WEIGHT_DECAY = 0.0      # L2 weight decay
	LEARN_UPDATES = 10      # number of learning updates
	UPDATE_EVERY = 20       # how often to update the network
```

## Results

![Scores](https://github.com/kumarakshay121/udacity_deepRL/blob/master/p2_continuous-control/scores.png)

The agent was able to solve the environment is 71 episodes, by maintaining an average score of 30.0+ across 20 agents and 100 consecutive episodes.

## Future Work

The implementation of [Distributed Distributional Deep Deterministic Policy Gradient](https://openreview.net/pdf?id=SyZipzbCb) (D4PG) might offer significant improvements over DDPG. D4PG uses a distributional version of critic which models randomness due to instric factors such as inherent uncertainty of function approximation in continuous environments. This distributional update leads to better gradients and stable learning. Moreover, D4PG also uses N-step returns when estimating TD error and a prioritized experience replay to provide higher weightage to experiences that are more important for learning. It might be interesting to compare the results of D4PG against DDPG.
