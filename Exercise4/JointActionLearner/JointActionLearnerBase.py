#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import itertools
import argparse
import numpy as np
from collections import defaultdict
from math import inf

class JointQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
		super(JointQLearningAgent, self).__init__()

		self.initLearningRate = learningRate
		self.discountFactor = discountFactor
		self.initEpsilon = epsilon
		self.initVals = initVals
		self.numTeammates = numTeammates

		self.Q = defaultdict(lambda: initVals)
		self.totalNumTimesteps = 0

		# dict for each other agent, counting the number of times they take an action
		self.model_counts = [defaultdict(lambda: initVals)]*numTeammates

	def setExperience(self, state, action, oppoActions, reward, status, nextState):
		# update model counts
		self.totalNumTimesteps += 1
		for i, a in enumerate(oppoActions):
			# print(self.model_counts[i])
			self.model_counts[i][(state, a)] += 1

		# print("Action: {}".format(action))
		# print("opp: {}".format(oppoActions))
		self.sample = (state, action, oppoActions, reward, nextState, status)

	def computeSum(self, state, action):
		# computes expected value of us taking an action from a state
		sum_value = 0
		# itertools product gives all possible joint actions of other agents
		for jointOppAcc in itertools.product(self.possibleActions, repeat=self.numTeammates):
			# print("joint")
			# print(jointOppAcc)
			prob = 1
			for opp in range(self.numTeammates):
				# print(jointOppAcc[opp])
				prob *= self.model_counts[opp][(state, jointOppAcc[opp])]/self.totalNumTimesteps
			# always assume your action goes first
			joint_a = tuple([action] + list(jointOppAcc))
			q = self.Q[(state, joint_a)]
			sum_value += prob * q
		return sum_value

	def learn(self):
		S, A, OA, R, Sp, status = self.sample

		# compute greedy action from next state
		values = []

		for action in self.possibleActions:
			value = self.computeSum(S, action)
			values.append(value)

		greedy_action_value = max(values)

		# if next state is actually a terminal state, then value is 0
		if status != "IN_GAME":
			greedy_action_value = 0

		# form join action with our action in front
		joint_a = tuple([A] + OA)
		# print(joint_a)

		# update Q
		oldQ = self.Q[(S, joint_a)]
		self.Q[(S,joint_a)] += self.learningRate*(R + \
				self.discountFactor*greedy_action_value - self.Q[(S,joint_a)])

		return self.Q[(S,joint_a)]  - oldQ

	def act(self):
		if np.random.rand() < 1 - self.epsilon:
			# choose greedy action
			greedy_actions = []
			greedy_action_value = -inf

			for action in self.possibleActions:
				action_value = self.computeSum(self.state, action)
				if action_value == greedy_action_value:
					greedy_actions.append(action)
				elif action_value > greedy_action_value:
					greedy_actions = [action]
					greedy_action_value = action_value

			action = random.sample(greedy_actions, 1)[0]
		else:
			# choose random action with uniform probability
			action = random.sample(self.possibleActions, 1)[0]
		return action

	def setEpsilon(self, epsilon) :
		self.epsilon = epsilon

	def setLearningRate(self, learningRate) :
		self.learningRate = learningRate

	def setState(self, state):
		self.state = state

	def toStateRepresentation(self, rawState):
		return tuple(tuple(tuple(q) for q in s) for s in rawState)

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		epsilon = max(0, 1.0-episodeNumber/45000)
		if episodeNumber < 5000:
			epsilon = 1.0
		if episodeNumber > 45000:
			epsilon = 0.0
		# elif episodeNumber < 5000:
		# 	epsilon = 1.0
		learningRate = max(0.05, 0.3 * epsilon) # 0.47 40000
		# learningRate = 0.3 * epsilon # 0.3
		# learningRate = max(0.05, 0.15 * epsilon) # 0.41
		# learningRate = 0.5 * epsilon

		return learningRate, epsilon

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	numAgents = args.numAgents
	numEpisodes = args.numEpisodes
	for i in range(numAgents):
		agent = JointQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0, numTeammates=args.numAgents-1)
		agents.append(agent)

	numEpisodes = numEpisodes
	numTakenActions = 0

	statusHistory = []

	for episode in range(numEpisodes):
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()

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
				agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())

			# print("1---{}".format(actions))
			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			for agentIdx in range(args.numAgents):
				oppoActions = actions.copy()
				# print("2---{}".format(oppoActions))
				del oppoActions[agentIdx]
				# print("3---{}".format(oppoActions))
				# print("4---{}".format(actions[agentIdx]))
				agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions,
					reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()

			observation = nextObservation

		statusHistory.append(status[0])
		# print(status)
		print(episode)
		print("-----\nGOAL PROPORTION: {}\n------".format(sum(1 for x in statusHistory if x == "GOAL")/len(statusHistory)))

	print(statusHistory)
	print(sum(1 for x in statusHistory[-5000:] if x == 'GOAL')/5000)
