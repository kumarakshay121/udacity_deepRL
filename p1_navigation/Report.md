# Project 1 Report (Navigation)

## Introduction

The objective of this project is to train an agent to navigate (and collect bananas!) in a large, square world.

## Method

To solve the problem, a vanilla DQN algorithm was chosen, which was introduced by [DeepMind](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). DQN is an effective value-based algorithm, which is proven to perform well in evironments with discrete state and actions spaces. It uses a deep neural network as a function approximator to estimate a Q-value function given a set of discrete states. The agent learns by observing the change in game score (reward) at the end of each episode. At each time step, the action that maximizes the Q-value is chosen.

Characteristic techniques of DQN that help train an agent better:
- **Experience replay**: A replay buffer is used to store previous experiences in order to sample and reuse them for training. Random sampling from a replay buffer breaks any harmful correlations between consecutive experiences and prevents action values from oscillating or diverging catastrophically. Moreover, the replay buffer is a more efficient use of past experiences as it provides the possibility to use experiences multiple times and recall rare occurences.
- **Fixed Q targets**: The parameters of temporal difference (TD) target is fixed for a certain number of steps, allowing the agent to train more stably since this decouples the learning parameters from the target parameters.

**Network Architecture**

The network consists of three fully-connected layers.

	hidden layer 1: 64 units + Leaky RELU activation		
	hidden layer 2: 64 units + Leaky RELU activation		
	output layer: 4 units (action size)

**Hyperparameters**

	BUFFER_SIZE = int(1e5)  # replay buffer size
	BATCH_SIZE = 64         # minibatch size
	GAMMA = 0.99            # discount factor
	TAU = 1e-3              # for soft update of target parameters
	LR = 5e-4               # learning rate 
	UPDATE_EVERY = 4        # how often to update the network

## Results

![Scores]()

The agent was able to solve the environment is 380 episodes, by maintaining an average score of 13.0+ across 100 consecutive episodes.

## Future Work

The project was simple enough to be solved using a vanilla DQN method. However, in general, a vanilla DQN suffers from overestimation of Q-values which can be addressed using Double DQNs. Also, the current implementation of experience replay cannot differentiate experiences that are more important for learning from the ones that are less important. This can be achieved using Prioritized Experience Replay. It might be interesting to compare the results of these different techniques over the vanilla DQN algorithm.
