#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
from collections import defaultdict

class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
		super(WolfPHCAgent, self).__init__()

		self.initLearningRate = learningRate
		self.discountFactor = discountFactor
		self.winDelta = winDelta
		self.loseDelta = loseDelta
		self.initVals = initVals

		self.Q = defaultdict(lambda: initVals)
		self.policy = defaultdict(lambda: 1/(len(self.possibleActions)))

		self.averagePolicy = defaultdict(lambda: 0)

		self.totalNumTimesteps = 0
		self.C = defaultdict(lambda: 0)


	def setExperience(self, state, action, reward, status, nextState):
		self.sample = (state, action, reward, nextState)

	def learn(self):
		# returns change (value after update - value before) of q(s,a)
		s, a, r, sp = self.sample

		oldQ = self.Q[(s,a)]
		self.Q[(s,a)] += self.learningRate * (r +  \
			self.discountFactor*max(self.Q[(sp,ap)] for ap in self.possibleActions) \
			- self.Q[(s,a)])

		return 	self.Q[(s,a)] - oldQ

	def act(self):
		# select action according to policy
		probabilities = [self.policy[(self.state,a)] for a in self.possibleActions]
		action = np.random.choice(self.possibleActions, 1, p=probabilities)[0]
		return action

	def calculateAveragePolicyUpdate(self):
		# increment counts
		self.totalNumTimesteps += 1
		self.C[self.state] += 1

		# update average policy
		for action in self.possibleActions:
			self.averagePolicy[(self.state, action)] +=  \
				(1/self.C[self.state])*(self.policy[(self.state, action)]
						-self.averagePolicy[(self.state, action)])

		return [self.averagePolicy[(self.state, a)] for a in self.possibleActions]

	def calculatePolicyUpdate(self):
		# find the suboptimal actions
		qmax = max(self.Q[(self.state, a)] for a in self.possibleActions)
		suboptimalActions = [a for a in self.possibleActions \
							if self.Q[(self.state, a)] != qmax]
		optimalActions = [a for a in self.possibleActions \
							if self.Q[(self.state, a)] == qmax]

		# decide the learning rate to use
		delta = None
		if sum(self.policy[(self.state, a)]*self.Q[(self.state, a)] for a in self.possibleActions) >= \
			sum(self.averagePolicy[(self.state, a)]*self.Q[(self.state, a)] for a in self.possibleActions):
			# winning
			delta = self.winDelta
		else:
			delta = self.loseDelta

		# update probability of suboptimal actions
		pmoved = 0
		for a in suboptimalActions:
			pmoved += min(delta/len(suboptimalActions), self.policy[(self.state, a)])
			self.policy[(self.state, a)] -= min(delta/len(suboptimalActions), self.policy[(self.state, a)])

		# update probability of optimal actions
		for a in optimalActions:
			self.policy[(self.state, a)] += pmoved/(len(optimalActions))

		return [self.policy[(self.state, a)] for a in self.possibleActions]

	def toStateRepresentation(self, state):
		return tuple(tuple(tuple(q) for q in s) for s in state)

	def setState(self, state):
		self.state = state

	def setLearningRate(self,lr):
		self.learningRate = lr

	def setWinDelta(self, winDelta):
		self.winDelta = winDelta

	def setLoseDelta(self, loseDelta):
		self.loseDelta = loseDelta

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		learningRate = max(0.05, 1.0-episodeNumber/40000) #  reaches 90 in last 5000
		return self.loseDelta, self.winDelta, learningRate

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	numOpponents = args.numOpponents
	numAgents = args.numAgents
	MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents, visualize=False)

	agents = []
	for i in range(args.numAgents):
		agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.01, loseDelta=0.1)
		agents.append(agent)

	statusHistory = []

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	for episode in range(numEpisodes):
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()

		while status[0]=="IN_GAME":
			for agent in agents:
				loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
				agent.setLoseDelta(loseDelta)
				agent.setWinDelta(winDelta)
				agent.setLearningRate(learningRate)
			actions = []
			perAgentObs = []
			agentIdx = 0
			for agent in agents:
				obsCopy = deepcopy(observation[agentIdx])
				perAgentObs.append(obsCopy)
				agent.setState(agent.toStateRepresentation(obsCopy))
				actions.append(agent.act())
				agentIdx += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			agentIdx = 0
			for agent in agents:
				agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx],
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agent.learn()
				agent.calculateAveragePolicyUpdate()
				agent.calculatePolicyUpdate()
				agentIdx += 1

			observation = nextObservation

		statusHistory.append(status[0])
		print(episode)
		print("-----\nGOAL PROPORTION: {}\n------".format(sum(1 for x in statusHistory if x == "GOAL")/len(statusHistory)))

	print(statusHistory)
	print(sum(1 for x in statusHistory[-5000:] if x == 'GOAL')/5000)
