#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
from collections import defaultdict
import numpy as np
import random
from math import inf

class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon=1.0, initVals=0.0):
		super(SARSAAgent, self).__init__()
		self.initLearningRate = learningRate
		self.discountFactor = discountFactor
		self.initEpsilon = epsilon
		self.initVals = initVals

		self.reset()
		self.Q = {}

	def learn(self):
		S, A, R, Sp, Ap = self.sarsa

		# to avoid key errors on new state action pairs
		if (S, A) not in self.Q:
			self.Q[(S, A)] = self.initVals
		if (Sp, Ap) not in self.Q:
			self.Q[(Sp, Ap)] = self.initVals

		oldq = self.Q[(S, A)]

		self.Q[(S, A)] = self.Q[(S, A)]  + self.learningRate * \
			(R + self.discountFactor * self.Q[(Sp, Ap)] - self.Q[(S, A)])

		# print("Updating ({}, {}) from {} to {}".format(S, A, oldq, self.Q[(S,A)]))
		return self.Q[(S, A)] - oldq

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

	def setState(self, state):
		self.state = state

	def setExperience(self, state, action, reward, status, nextState):
		if self.prevState is not None:
			# the tuple to next be used for update
			self.sarsa = (self.prevState, self.prevAction, self.prevReward, state, action)

		# record for our next update
		self.prevAction = action
		self.prevState = state
		self.prevReward = reward
		# agent state is set by setState later

		if reward is not None:
			self.totalReward += self.discountFactor**self.stepsTaken * reward

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		epsilon = max(0, 1.0-episodeNumber/4500)
		if episodeNumber >= 4500:
			epsilon = 0.0
		learningRate = max(0.001, epsilon*0.2)

		return (learningRate, epsilon)

	def toStateRepresentation(self, state):
		return tuple(state)

	def reset(self):
		self.sarsa = None
		self.state = None
		self.stepsTaken = 0
		self.prevState = None
		self.prevAction = None
		self.prevReward = None
		self.learningRate = self.initLearningRate
		self.epsilon = self.initEpsilon
		self.totalReward = 0

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=5000)

	args=parser.parse_args()

	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(0.1, 0.99)




	statusHistory = []
	rewardHistory = []

	# Run training using SARSA
	numTakenActions = 0
	for episode in range(numEpisodes):
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)

			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			# print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))

			if not epsStart :
				# pass
				agent.learn()
			else:
				epsStart = False

			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()

		print("----------\nSTATUS: {}\n--------".format(status))
		print("EPISODE: {}".format(episode))
		print("EPSILON: {}".format(epsilon))
		print("LEARNING: {}".format(learningRate))

		statusHistory.append(status)
		# print("-----\nSTATUS HISTORY: {}\n------".format(statusHistory))

		rewardHistory.append(agent.totalReward)
		print("MEAN: {}\n------".format(np.mean(rewardHistory)))

	# print("-----\nREWARD HISTORY: {}\nMEAN: {}\n------".format(rewardHistory, np.mean(rewardHistory)))

	# reaches 0.8 in last 500
	print("-----\nFINAL GOAL PROPORTION: {}\n------".format(sum(1 for x in statusHistory[-500:] if x == 1)/500))

	print("\n\n\n")

	# print(agent.Q)
