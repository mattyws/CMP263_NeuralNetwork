import random
import sys
import math

import pandas as pd
import numpy as np


class NeuralNetwork:

    def __init__(self, layers=[], weights=None, lamb=.01, K=50):
        self.layers = layers
        self.weights = weights
        self.lamb = lamb
        self.K = K

    def train(self, X, Y, total_labels):
        aux = [len(X.columns)]
        aux.extend(self.layers)
        aux.append(total_labels)
        self.layers = aux
        if self.weights is None:
            self.generate_random_weights()


        #self.mini_batch(X, Y)


    def backpropagation(self, X, Y, alphas):

        pass


    def mini_batch(self, X, Y):
        for i in range(0, int( math.ceil(len(X)/self.K)) ):
            max = 0
            if ( (i + 1) * self.K) > len(X) :
                max = len(X)
            else:
                max = ( (i + 1) * self.K)
            self.backpropagation( X, Y, self.k_batch(X, i*self.K, max) )


    def k_batch(self, X, min, max):
        sum = []
        for i in range(min, max):
            sum = np.sum([sum, self.forward_propagation( X.iloc[i] )], axis = 0 )
        return sum / (max - min)



    def generate_random_weights(self):
        self.weights = []
        for i in range(1, len(self.layers)):
            weights = []
            for j in range(int(self.layers[i])):
                weights.append([random.uniform(-1, 1) for x in range(int(self.layers[i - 1]) + 1)])
            self.weights.append(weights)

    def function_g(self, values):
        return (1.0 / (1.0 + np.exp(-values) ) )

    def result_matrix(self, weights, neurons_values ):
        neurons_values = np.array(neurons_values)
        weights = np.array(weights)
        aux = [1.0]
        aux.extend(neurons_values)
        aux = np.array(aux)[np.newaxis]

        return np.matmul(weights, aux.T)


    def forward_propagation(self, x):
        """
        :param x: instância (linha) da base de dados
        :return: alphas dos neurônios calculados para essa instância
        """
        alphas = []
        alphas.append(self.function_g(self.result_matrix(self.weights[0], x)))
        for i in range(1, len(self.weights)):
            alphas.append(self.function_g(self.result_matrix(self.weights[i], alphas[i - 1])))
        return alphas



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
    data = pd.read_csv(dataset_file, header=None, sep=',')
    # Cria colunas
    data.columns = ["TipoVinho", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
                    "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
                    "OD280/OD315 of diluted wines", "Proline"]

    labels = data[data.columns[0]]
    total_labels = len(labels.unique() )
    data = data.drop(data[data.columns[0:1]], axis=1)


    def normalize(data):
        data = (data - data.mean() ) / data.std()
        return data

    def one_hot_encode(labels):
        total_classe = len(labels.unique() )
        new_dict = dict()
        for i in range(0, total_classe ):
            new_dict[labels.unique()[i]] = np.zeros(total_classe)
            new_dict[labels.unique()[i]][i] = 1
        # print(new_dict)
        new_labels = []
        for i in labels:
            new_labels.append(new_dict[i])
        new_labels = pd.Series(new_labels)
        print(new_labels.unique())
        print(new_labels)
        return new_labels


    data = normalize(data)
    one_hot_encode(labels)
    nn = NeuralNetwork(layers=network_content[2:-1], lamb=network_content[0])
    nn.train(data, labels, total_labels)
