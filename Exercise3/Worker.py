import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import numpy as np
from math import inf


def train(idx, port, target_network, value_network, lock, counter, num_episodes=16000, name=""):

    print("Starting a worker {}".format(port))

    # port = 2207
    seed = 2207
    hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfoEnv.connectToServer()

    episodeNumber = 0
    epsilon = 1
    discountFactor = 0.99

    I_async_update = 5
    I_target = 10000

    goals = 0
    paramSaves = 0
    lastSaved = 0

    hard_update(target_network, value_network)

    optimizer = optim.Adam(value_network.parameters(), lr=1e-5)
    optimizer.zero_grad()

    t = 0  # local timestep counter

    if idx == 0:  # first thread keeps track of stats
        stats = []

    # run for certain number of timesteps
    while t < num_episodes * 500:

        timesteps_to_goal = 0  # for measuring performance
        total_reward = 0  # accumulated reward (without discounting)

        status = 0
        observation = hfoEnv.reset()
#        print(observation.shape)

        # linearly decrease epsilon
        epsilon = max(0.0, (22000-episodeNumber)/22000)

        while status == 0:

            # EPSILON GREEDY - TAKE AN ACTION

            if np.random.rand() < epsilon:
                # choose a random action
                action = np.random.choice(range(len(hfoEnv.possibleActions)))
            else:
                # choose greedy action
                lock.acquire()
                values = value_network(torch.Tensor(observation)).detach().numpy()
                action = np.argmax(values)
                lock.release()

            newObservation, reward, done, status, info = hfoEnv.step(hfoEnv.possibleActions[action])

            total_reward += reward

            # keep track of goals scored
            if reward >= 50.0:
                goals += 1

            # COMPUTE TARGET VALUE
            lock.acquire()
            target_value = computeTargets(reward, [newObservation],
                                          discountFactor, done, target_network)

            prediction = computePrediction([observation], action, value_network)

            loss = 0.5 * (prediction - target_value.detach())**2

            # accummulate gradient
            loss.backward()
            lock.release()

            observation = newObservation

            # update local counter t
            t += 1
            timesteps_to_goal += 1

            # update global counter T
            lock.acquire()
            counter.value += 1

            # update target network
            if counter.value % I_target == 0:
                hard_update(target_network, value_network)

            # only the first worker saves the model (every 1 mil)
            if idx == 0 and counter.value >= 1000000 + lastSaved:
                lastSaved = counter.value
                print("saving model")
                paramSaves += 1
                path = "{}_params_{}".format(name, paramSaves)
                saveModelNetwork(value_network, path)

            # update value network and zero gradients
            if t % I_async_update == 0 or done:
                print("Doing async update")
                optimizer.step()
                optimizer.zero_grad()

            lock.release()

            if done:
                if idx == 0:
                    timesteps_to_goal = timesteps_to_goal if status == 1 else 500
                    stats.append(timesteps_to_goal)
                    mean = np.mean(stats)  # mean ep length
                    # output things to a csv for monitoring
                    print("{}, {}, {}, {}, {}, {}, {}, {}".format(episodeNumber, t, mean, goals, epsilon, timesteps_to_goal, status, total_reward), file=open("{}experiment.csv".format(name), "a"))
                episodeNumber += 1

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
    # target value computation for Q-Learning; should be usable for any target network arch
    if done:
        target_value = torch.Tensor([reward])
    else:
        # compute the best action from next state according to targetNetwork
        values = targetNetwork(torch.Tensor(nextObservation)).detach().numpy()
        greedy_value = np.max(values)
        target_value = torch.Tensor([reward]) + discountFactor * greedy_value

    return target_value

def computePrediction(state, action, valueNetwork):
    # implement a single call for forward computsation of value Q-network
    # again, arch agnostic
    prediction = valueNetwork(torch.Tensor(state))
    return prediction[0][action]


# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
    torch.save(model.state_dict(), strDirectory)
