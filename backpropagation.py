import random
import sys
import math

import pandas as pd
import numpy as np

import helper
from neuralnetwork import NeuralNetwork

if "__main__":
    nseed = 100
    random.seed(nseed)
    network = sys.argv[1]
    network_content = []
    with open(network, 'r') as network_file_handler:
        for line in network_file_handler.readlines():
            network_content.append(float(line.strip()))
    initial_weights_file = sys.argv[2]
    initial_weights = []
    with open(initial_weights_file, 'r') as initial_weights_file_handler:
        for line in initial_weights_file_handler.readlines():
            weights = []
            for neurons_weights in line.strip().split(';'):
                neuron_weights = []
                for weight in neurons_weights.split(','):
                    neuron_weights.append(float(weight))
                weights.append(neuron_weights)
            initial_weights.append(weights)
    dataset_file = sys.argv[3]
    # WINE
    X = pd.read_csv(dataset_file, header=None, sep=',')
    # TODO: colocar no relatório que nosso script considera última coluna como classes
    labels = X[X.columns[0]]
    total_labels = len(labels.unique() )
    X = X.drop(columns=X.columns[0])

    X = helper.normalize(X, X.mean(), X.std())
    # TODO: nosso script considera que as classes já estão numéricas
    labels = helper.one_hot_encode(labels)
    nn = NeuralNetwork(layers=network_content[2:-1], weights=initial_weights,
                       lamb=0.0, max_iter=10000, learning_rate=0.001,
                       threshold=0.00001)
    nn.check_gradient(X, labels, total_labels)
