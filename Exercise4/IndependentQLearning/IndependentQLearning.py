#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
import numpy as np
from math import inf

class IndependentQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()

		self.initLearningRate = learningRate
		self.discountFactor = discountFactor
		self.initEpsilon = epsilon
		self.initVals = initVals

		self.Q = {}
		self.stepsTaken = 0

	def setExperience(self, state, action, reward, status, nextState):
		self.sample = (state, action, reward, nextState, status)

	def learn(self):
		S, A, R, Sp, status = self.sample

		# compute greedy action from next state
		greedy_action_value = -inf
		for action in self.possibleActions:
			if (Sp, action) not in self.Q:
				self.Q[(Sp, action)] = self.initVals
			action_value = self.Q[(Sp, action)]
			if action_value >= greedy_action_value:
				greedy_action_value = action_value

		if (S, A) not in self.Q:
			self.Q[(S, A)] = self.initVals

		# if next state is terminal, then value is 0
		if status != "IN_GAME":
			greedy_action_value = 0

		oldQ = self.Q[(S, A)]

		self.Q[(S,A)] += self.learningRate*(R + self.discountFactor*greedy_action_value - self.Q[(S,A)])

		return self.Q[(S,A)]  - oldQ

	def act(self):
		if np.random.rand() < 1 - self.epsilon:
			# choose greedy action
			greedy_actions = []
			greedy_action_value = -inf

			for action in self.possibleActions:

				if (self.state, action) not in self.Q:
					self.Q[(self.state, action)] = self.initVals
				action_value = self.Q[(self.state, action)]

				if action_value == greedy_action_value:
					greedy_actions.append(action)
				elif action_value > greedy_action_value:
					greedy_actions = [action]
					greedy_action_value = action_value

			action = random.sample(greedy_actions, 1)[0]
		else:
			# choose random action with uniform probability
			action = random.sample(self.possibleActions, 1)[0]

		self.stepsTaken += 1
		return action

	def toStateRepresentation(self, state):
		return tuple(tuple(tuple(q) for q in s) for s in state)

	def setState(self, state):
		self.state = state

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		epsilon = max(0, 1.0-episodeNumber/45000)
		learningRate = epsilon*0.1
		if episodeNumber > 45000:
			epsilon = 0.0

		# reaches 90% in last 5000

		return learningRate, epsilon

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents, visualize=False)
	agents = []
	for i in range(args.numAgents):
		agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0

	statusHistory = []

	for episode in range(numEpisodes):
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		totalReward = 0.0
		timeSteps = 0

		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())
			numTakenActions += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)

			for agentIdx in range(args.numAgents):
				agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx],
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()

			observation = nextObservation

		statusHistory.append(status[0])
		# print(status)
		print(episode)
		print("-----\nGOAL PROPORTION: {}\n------".format(sum(1 for x in statusHistory if x == "GOAL")/len(statusHistory)))

	# print(statusHistory)
	print(sum(1 for x in statusHistory[-5000:] if x == 'GOAL')/5000)
