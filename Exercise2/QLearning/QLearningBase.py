#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import random
from math import inf

class QLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()

		self.initLearningRate = learningRate
		self.discountFactor = discountFactor
		self.initEpsilon = epsilon
		self.initVals = initVals

		self.Q = {}
		self.reset()

	def learn(self):
		S, A, R, Sp, status = self.sample

		# print(Sp[0])
		# print(status)

		# compute greedy action from next state
		greedy_action_value = -inf
		for action in self.possibleActions:
			if (Sp, action) not in self.Q:
				self.Q[(Sp, action)] = self.initVals
			action_value = self.Q[(Sp, action)]
			if action_value > greedy_action_value:
				greedy_action_value = action_value

		if status != 0:
			# terminal states have zero value
			greedy_action_value = 0

		if (S, A) not in self.Q:
			self.Q[(S, A)] = self.initVals

		oldQ = self.Q[(S, A)]

		self.Q[(S,A)] += self.learningRate*(R + self.discountFactor*greedy_action_value - self.Q[(S,A)])

		# print("Update: {} to {}".format((S,A), self.Q[(S,A)]))

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
			# print("Choosing greedy action: {} ({}) from {}".format(action, greedy_action_value, greedy_actions))
		else:
			# choose random action with uniform probability
			action = random.sample(self.possibleActions, 1)[0]
			# print("Choosing random action: {}".format(action))

		self.stepsTaken += 1
		return action

	def toStateRepresentation(self, state):
		return tuple(state)

	def setState(self, state):
		self.state = state

	def setExperience(self, state, action, reward, status, nextState):
		self.sample = (state, action, reward, nextState, status)

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def reset(self):
		self.stepsTaken = 0
		self.state = None

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		# learningRate = self.initLearningRate
		if episodeNumber < 250:
			epsilon = 1.0
		else:
			epsilon = max(0, 1.0-episodeNumber/4750)
		learningRate = 0.2 * epsilon

		return (learningRate, epsilon)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=5000)

	args=parser.parse_args()

	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes


	statusHistory = []
	rewardHistory = []

	# Run training using Q-Learning
	numTakenActions = 0
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)

			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			update = agent.learn()

			observation = nextObservation

		# print(agent.Q)

		statusHistory.append(status)
		# print("-----\nSTATUS HISTORY: {}\n------".format(statusHistory))

		print(episode)
		print("-----\nGOAL PROPORTION: {}\n------".format(sum(1 for x in statusHistory if x == 1)/len(statusHistory)))

	print(statusHistory)
	print("-----\nGOAL PROPORTION: {}\n------".format(sum(1 for x in statusHistory[-500:] if x == 1)/500))
	print(agent.Q)
