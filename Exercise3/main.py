#!/usr/bin/env python3
# encoding utf-8

# "This script should contain the necessary processes to run your asynchronous agent. "

import torch.multiprocessing as mp
import torch
import argparse
from Worker import train
from Networks import ValueNetwork

# Use this script to handle arguments and
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2648, help="Base server port")
    parser.add_argument('--seed', type=int, default=2207,
	                  help="Python randomization seed; uses python default if 0 or not given")
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--num_episodes', type=int, default=20000)
    parser.add_argument('--name', type=str, help="name to be used for experiment logs",default="")

    args = parser.parse_args()

    # make the shared networks
    value_network = ValueNetwork()
    target_network = ValueNetwork()

    value_network.share_memory()
    target_network.share_memory()

	# Example on how to initialize global locks for processes
	# and counters.

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    processes = []

    # how to asynchronously call multiple instances of train
    for idx in range(0, args.num_processes):
        # trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter)
        port = args.port + 51*idx
        print(port)
        p = mp.Process(target=train, args=(idx, port, value_network, target_network, lock, counter, args.num_episodes, args.name))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
