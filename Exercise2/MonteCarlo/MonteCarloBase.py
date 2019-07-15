#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import random
from math import inf

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.discountFactor = discountFactor
		self.epsilon = epsilon
		self.initVals = initVals

		self.numEpisodes = 1
		self.reset()

		self.Q = {}  # empty dict
		self.returns = {}

		self.totalReward = 0

	def learn(self):
		G = 0
		updateValues = []

		for i in range(len(self.experience)):
			t = len(self.experience) - 1 - i  # going backwards through ep
			state, action, reward = self.experience[t]

			G = self.discountFactor * G + reward

			# check if this is the first visit
			if self.firstVisitTimes[(state, action)] == t:

				# append G to returns for this state
				if (state, action) not in self.returns:
					self.returns[(state, action)] = []
				self.returns[(state, action)].append(G)

				if (state, action) not in self.Q:
					self.Q[(state, action)] = self.initVals

				# print("{} is being updated from {}".format((state, action), self.Q[(state, action)]))
				self.Q[(state, action)] = np.mean(self.returns[(state, action)])
				# print("to {}".format(self.Q[(state, action)]))

				updateValues.append(self.Q[(state, action)])

		updateValues.reverse()

		return (self.Q, updateValues)

	def toStateRepresentation(self, state):
		# state is received as a list [(0,2),(2,2)]
		return tuple(state)

	def setExperience(self, state, action, reward, status, nextState):
		# record (s,a,r) for each timestep
		self.experience.append((state, action, reward))

		# also record when we first visit each (state, action)
		if (state, action) not in self.firstVisitTimes:
			self.firstVisitTimes[(state, action)] = len(self.experience)-1

		# no reward at end of episode
		if reward is not None:
			self.totalReward += self.discountFactor**self.stepsTaken * reward

	def setState(self, state):
		self.state = state

	def reset(self):
		self.state = None
		self.firstVisitTimes = {}
		self.stepsTaken = 0
		self.experience = []
		self.totalReward = 0

	def act(self):
		# e greedy policy
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

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		if episodeNumber >= 4500:
			return 0.0
		return max(0, 1.0-episodeNumber/4500)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes
	numTakenActions = 0

	statusHistory = []
	rewardHistory = []
	meanRewardHistory = []

	# Run training Monte Carlo Method
	for episode in range(numEpisodes):
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation

		agent.learn()
		print("----------\nSTATUS: {}\n--------".format(status))
		print("EPISODE: {}".format(episode))
		print("EPSILON: {}".format(epsilon))

		# print("Q TABLE:")
		# print(agent.Q)
		# print()

		statusHistory.append(status)
		# print("-----\nSTATUS HISTORY: {}\n------".format(statusHistory))

		print("-----\nGOAL PROPORTION: {}\n------".format(sum(1 for x in statusHistory if x == 1)/len(statusHistory)))

		rewardHistory.append(agent.totalReward)

		meanRewardHistory.append(np.mean(rewardHistory))

	# print("----\nMEAN REWARD HISTORY: {}\n-----".format(meanRewardHistory))

	print("-----\nGOAL PROPORTION: {}\n------".format(sum(1 for x in statusHistory[-500:] if x == 1)/500))
	# reaches 0.8 in last 500

	# print("Q TABLE:")
	# print(agent.Q)
