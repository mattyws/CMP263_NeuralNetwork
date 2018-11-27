import random
import sys
import math

import pandas as pd
import numpy as np

import helper
from neuralnetwork import NeuralNetwork

if "__main__":
    random.seed(10)
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
    if len(initial_weights) != 0:
        # generate_random weights
        pass
    dataset_file = sys.argv[3]
    # WINE
    X = pd.read_csv(dataset_file, header=None, sep=',')
    # Cria colunas
    X.columns = ["TipoVinho", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
                    "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
                    "OD280/OD315 of diluted wines", "Proline"]

    labels = X[X.columns[0]]
    total_labels = len(labels.unique() )
    X = X.drop(X[X.columns[0:1]], axis=1)

    X = helper.normalize(X)
    labels = helper.one_hot_encode(labels)
    nn = NeuralNetwork(layers=network_content[2:-1], lamb=network_content[0], max_iter=10000, learning_rate=0.001,
                       threshold=0.00001)
    nn.train(X, labels, total_labels)
    predicted = nn.predict(X)
    print(predicted)
