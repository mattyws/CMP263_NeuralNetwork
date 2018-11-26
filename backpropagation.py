import random
import sys
import math

import pandas as pd
import numpy as np


class NeuralNetwork:

    def __init__(self, layers=[], weights=None, lamb=.01, K=50, learning_rate=0.00001, max_iter=666, threshold=0.000001):
        self.layers = layers
        self.weights = weights
        self.lamb = lamb
        self.K = K
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold

    def train(self, X, Y, total_labels):
        aux = [len(X.columns)]
        aux.extend(self.layers)
        aux.append(total_labels)
        self.layers = aux
        if self.weights is None:
            self.generate_random_weights()
        J = np.inf
        for i in range(self.max_iter):
            print(i)
            aux_J = self.backpropagation(X, Y)
            if i % 100 == 0:
                print(abs(J - aux_J))
            if abs(J - aux_J) <= self.threshold:
                print("Gatilho de parada", abs(J - aux_J))
                break
            J = aux_J

    def predict(self, X):
        if X is not None and len(X) != 0:
            if type(X) == type(pd.Series()):
                alphas = self.forward_propagation(X)
                return self.transform_to_class(alphas[len(alphas)-1])
            else:
                # If has more than one row, do the predicting for all data given
                classes = []
                for index, x in X.iterrows():
                    alphas = self.forward_propagation(x)
                    classes.append(self.transform_to_class(alphas[len(alphas)-1]))
                return classes

    def backpropagation(self, X, Y):
        sum_j = 0
        for i in range(0, int( math.ceil(len(X)/self.K)) ):
            max = 0
            if ( (i + 1) * self.K) > len(X) :
                max = len(X)
            else:
                max = ( (i + 1) * self.K)
            gradients, j = self.k_batch(X, Y, i*self.K, max)
            sum_j += j
            self.update_weights(gradients)
        S = 0
        for i in range(len(self.weights)):
            aux = np.array(self.weights[i])
            aux[:, 0] = 0
            S += np.sum(aux**2)
        # print("S", S)
        S *= ( self.lamb / (2*len(X)) )
        # print("Esse S", S)
        return (sum_j / len(X)) + S


    def k_batch(self, X, Y, min, max):
        #sum_alpha = []
        #sum_delta = []
        sum_gradient = []
        sum_j = 0
        for i in range(min, max):
            alphas = np.array(self.forward_propagation( X.iloc[i] ))
            #sum_alpha = np.sum([sum_alpha, alphas ], axis = 0 )
            deltas = np.array(self.backward_deltas(Y.iloc[i], alphas ))
            j = self.function_j(deltas[len(deltas) - 1])
            sum_j = sum_j + j
            # sum_delta = np.sum([sum_delta, deltas], axis = 0 )
            # print("============ k-batch ================")
            # print("alphas: ", alphas)
            # print("deltas: ", deltas )
            # print("============ k-batch ================")
            # self.weight_gradient(X.iloc[i], alphas, deltas)
            grad = self.weight_gradient(X.iloc[i], alphas, deltas)
            # print("grad: ", grad)
            #sum_gradient = np.sum( [sum_gradient, grad], axis = 0 )
            if len(sum_gradient) == 0:
                sum_gradient = grad
            else:
                sum_gradient = np.sum([sum_gradient, grad], axis = 0  )
        # print("sum: ", sum_gradient)

        #retorna gradiente médio
        return sum_gradient / (max - min), sum_j

    def weight_gradient(self, x, alphas, deltas):
        gradients = []
        aux = [1.0]
        aux.extend(x)
        aux = np.array(aux)[np.newaxis]
        # print("aux weight_gradient: ", aux)
        gradients.append(np.matmul(deltas[0], aux))
        for i in range(len(alphas)-1):
            gradients.append(np.matmul(deltas[i+1], alphas[i].T))
        # print("gradients antes do for", len(gradients[0][0]))
        for i in range(len(gradients)):
            aux = np.array(self.weights[i])
            aux[:, 0] = 0
            gradients[i] = gradients[i] + (self.lamb * aux)
        # print("gradients", gradients)
        return gradients

    def generate_random_weights(self):
        self.weights = []
        for i in range(1, len(self.layers)):
            weights = []
            for j in range(int(self.layers[i])):
                weights.append( [ random.uniform(-1, 1) for x in range(int(self.layers[i - 1]) + 1) ] )
            self.weights.append(np.array(weights))
        self.weights = np.array(self.weights)

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
            alphas.append( self.function_g(self.result_matrix(self.weights[i], alphas[i - 1])))
        new_alphas = []
        for i in range(len(alphas)-1):
            aux = [[1.0]]
            aux.extend( alphas[i] )
            new_alphas.append( np.array(aux))
        new_alphas.append(alphas[len(alphas)-1])
        #print("new alphas", new_alphas)
        return new_alphas

    def backward_deltas(self, y, alphas):
        deltas = []
        deltas.append(self.delta_output(alphas[len(alphas)-1], y ))
        for i in range(len(alphas)-1, 0, -1):
            #print("==================================================================")
            deltas = [self.delta_hidden(alphas[i-1], self.weights[i], deltas[0])] + deltas
        return deltas

    def delta_hidden(self, alpha, weights, deltas):
        delta = np.matmul(weights.T, deltas ) * alpha * (1 - alpha)
        delta = delta[1:]
        return delta

    def delta_output(self, alpha_output, real_output):
        #print("log alpha output", -real_output * np.log2(alpha_output.T))
        delta = np.array((-real_output * np.log2(alpha_output.T)) - (1 - real_output) * np.log2((1-alpha_output.T))).T
        #print("delta_output", delta)
        return delta

    def update_weights(self, gradients):
        self.weights -= (self.learning_rate * gradients)

    def transform_to_class(self, alphas):
        output = np.zeros(len(alphas))
        output[np.argmax(alphas)] = 1
        print(output)
        return output

    def function_j(self, output):
        return np.sum(output)


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
        return new_labels


    X = normalize(X)
    labels = one_hot_encode(labels)
    nn = NeuralNetwork(layers=network_content[2:-1], lamb=network_content[0], max_iter=10000, learning_rate=0.00000001,
                       threshold=0.00000001)
    nn.train(X, labels, total_labels)
    nn.predict(X)
