# Project 3 Report (Tennis)

## Introduction

The objective of this project is to train two agents to compete with each other in the game of tennis (Unity's [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment).

## Method

To solve the problem, a Multi-agent Deep Deterministic Policy Gradient (MADDPG) algorithm was chosen, which was introduced by OpenAI in the paper [Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf). This algorithm uses a centralized planning and decentralized execution framework, i.e., each agent has access to observations from all other agents during training but only it's own observations during testing. Each agent has it's own actor and critic. The actor uses a deep neural network as a function approximator to estimate a deterministic policy, i.e., it aims to estimate the best possible action for any given state. The critic uses a deep neural network as a function approximator to estimate a Q-value function and is trained using the actors best believed action as target.

Characteristic techniques of DDPG that help train an agent better:
- **Experience replay**: A replay buffer is used to store previous experiences in order to sample and reuse them for training. Random sampling from a replay buffer breaks any harmful correlations between consecutive experiences and prevents action values from oscillating or diverging catastrophically. Moreover, the replay buffer is a more efficient use of past experiences as it provides the possibility to use experiences multiple times and recall rare occurences.
- **Soft updates**: A soft update involves slowly blending the regular weights with the target network weights instead of copying them at once. This approach leads to faster convergence.

In this project, two agents were trained to play the game of tennis. Both agents shared a common replay buffer, which enable them to benefit from each other's experiences.

**Network Architecture**

- Actor
```bash
	hidden layer 1: 256 units + RELU activation
	hidden layer 2: 128 units + RELU activation
	output layer: 2 units (action size)
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
	BATCH_SIZE = 256        # minibatch size
	GAMMA = 0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR_ACTOR = 1e-4         # learning rate of the actor
	LR_CRITIC = 1e-3        # learning rate of the critic
	WEIGHT_DECAY = 0.0      # L2 weight decay
	LEARN_UPDATES = 2       # number of learning updates
	UPDATE_EVERY = 2        # how often to update the network
```

## Results

![Scores](https://github.com/kumarakshay121/udacity_deepRL/blob/master/p3_collab-compet/scores.png)

The algorithm was able to solve the environment is 1327 episodes, by maintaining an average score of +0.5 across both the agents and 100 consecutive episodes.

## Future Work

Currently, the algorithm takes a significant amount of time before it starts to learn. This can probably be improve by hyperparameter tuning. For example, introduce more noise in the beginning to encourage exploration. This might also lead to different play strategies, such as lifting the ball and then returning.

It might also be interesting to try multi-agent implementation of policy-based methods, such as Proximal Policy Optimization (PPO) or Trust Region Policy Optimization (TRPO). Since policy-based methods estimate the policy directly, they might prove to be faster than DDPG.

Also, a multi-agent implementation of [Distributed Distributional Deep Deterministic Policy Gradient](https://openreview.net/pdf?id=SyZipzbCb) (D4PG) might offer significant improvements over DDPG. D4PG uses distributional updates, which leads to better gradients and stable learning. Moreover, D4PG also uses N-step returns when estimating TD error and a prioritized experience replay to provide higher weightage to experiences that are more important for learning. It might be interesting to compare the results of multi-agent D4PG against MADDPG.
