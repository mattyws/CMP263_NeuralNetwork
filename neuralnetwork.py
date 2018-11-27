import math
import random

import numpy as np
import pandas as pd

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
        self.check_gradient(X, Y)
        # J = np.inf
        # for i in range(self.max_iter):
        #      aux_J = self.backpropagation(X, Y)
        #      if i % 100 == 0:
        #          print("J", J, "auxJ", aux_J)
        #          print("diff", abs(J - aux_J))
        #      if abs(J - aux_J) <= self.threshold:
        #          print("Gatilho de parada", abs(J - aux_J))
        #          break
        #      J = aux_J

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
            gradients, j, ignore = self.k_batch(X, Y, i*self.K, max)
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
            output_error = self.output_error(alphas[len(alphas)-1], Y.iloc[i])
            j = self.function_j(output_error)
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
        return sum_gradient / (max - min), sum_j, sum_gradient

    def output_error(self, alpha_output, real_output):
        error = np.array((-real_output * np.log2(alpha_output.T)) - (1 - real_output) * np.log2((1 - alpha_output.T))).T
        return error

    def check_gradient(self, X, Y):

        # calcula a estimativa numérica do gradiente
        numGrad = self.computeNumericalGradient(X, Y)

        # calcula a derivada do gradiente
        sum_gradient = []
        for i in range(0, len(X) ):
            alphas = np.array(self.forward_propagation(X.iloc[i]))
            deltas = np.array(self.backward_deltas(Y.iloc[i], alphas))
            grad = self.weight_gradient(X.iloc[i], alphas, deltas)
            if len(sum_gradient) == 0:
                sum_gradient = grad
            else:
                sum_gradient = np.sum([sum_gradient, grad], axis=0)
        #derGrad = sum_gradient / len(X)
        derGrad = sum_gradient


        sum = []
        sub = []
        for p in range(len(numGrad)):
            #print("numGrad[p].ravel()", numGrad[p].ravel())
            sum = np.concatenate( (sum, numGrad[p].ravel() + derGrad[p].ravel() ), axis = 0 )
            #print("sum: ", sum)
            sub = np.concatenate((sub, numGrad[p].ravel() - derGrad[p].ravel()), axis=0)

        print("numGrad:", numGrad)
        print("derGrad:", derGrad)
        #print("numGrad - derGrad", numGrad - derGrad)
        #print("numGrad + derGrad", numGrad + derGrad)
        diff = np.linalg.norm( sub ) / np.linalg.norm( sum )

        print("diff", diff)
        if diff < 10e-8:
            return True
        else:
            return False

    def computeNumericalGradient(self, X, Y):
        weights = np.array(self.weights)
        #perturb = np.zeros(weights.shape)
        #numGrad = np.zeros(weights.shape)

        perturb = []
        numGrad = []
        #print("len(self.layers)", len(self.layers))
        for i in range(1, len(self.layers)):
            p = []
            numG = []
            #print("self.layers[i]", self.layers[i])
            for j in range(int(self.layers[i])):
                p.append([0.0 for x in range(int(self.layers[i - 1]) + 1)])
                numG.append([0.0 for x in range(int(self.layers[i - 1]) + 1)])
                #print("p", p)
                #p.append( np.zeros(self.layers[i-1] ) )
            perturb.append(np.array(p))
            numGrad.append(np.array(numG))
        perturb = np.array(perturb)
        numGrad = np.array(numGrad)

        #print("perturb ", perturb)

        E = 10e-4

        for p in range(len(weights)):
            #print("weights[p]", weights[p])
            for q in range(len(weights[p])):
                # print("weights[p][q]", weights[p][q])
                for r in range(len(weights[p][q])):
                    perturb[p][q][r] = E
                    #print("perturb[p][q][i]", perturb[p][q][i])

                    #print("self.weights antes", self.weights[p][q][i])
                    self.weights = weights + perturb
                    #print("self.weights depois", self.weights[p][q][i])

                    J_pos = 0.0
                    for i in range(0, len(X)):
                        alphas = np.array(self.forward_propagation(X.iloc[i]))
                        output_error = self.output_error(alphas[len(alphas) - 1], Y.iloc[i])
                        J_pos += self.function_j(output_error)

                    self.weights = weights - perturb

                    J_neg = 0.0
                    for i in range(0, len(X)):
                        alphas = np.array(self.forward_propagation(X.iloc[i]))
                        output_error = self.output_error(alphas[len(alphas) - 1], Y.iloc[i])
                        # soma do erro da rede para cada instância
                        J_neg += self.function_j(output_error)

                    # faz a média dos J_pos e J_neg por que eles são a soma dos erros da rede para cada instância
                    numGrad[p][q][r] = (  J_pos - J_neg  ) / (2 * E)
                    #self.update_weights( )
                    # perturb tem que ser zerado de novo no final do loop
                    perturb[p][q][r] = 0.0

        # volta pesos para pesos originais
        self.weights = weights
        return numGrad

    def weight_gradient(self, x, alphas, deltas):
        gradients = []
        aux = [1.0]
        aux.extend(x)
        aux = np.array(aux)[np.newaxis]
        # print("aux weight_gradient: ", aux)
        gradients.append(np.matmul(deltas[0], aux))
        print("aux: ", aux)
        print("deltas[0]", deltas[0])
        print("gradients", gradients)
        for i in range(len(alphas)-1):
            gradients.append(np.matmul(deltas[i+1], alphas[i].T))
        # print("gradients antes do for", len(gradients[0][0]))
        # for i in range(len(gradients)):
        #     aux = np.array(self.weights[i])
        #     aux[:, 0] = 0
        #     gradients[i] = gradients[i] + (self.lamb * aux)
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
        return new_alphas

    def backward_deltas(self, y, alphas):
        deltas = []
        deltas.append(self.delta_output(alphas[len(alphas)-1], y ))
        for i in range(len(alphas)-1, 0, -1):
            deltas = [self.delta_hidden(alphas[i-1], self.weights[i], deltas[0])] + deltas
        return deltas

    def delta_hidden(self, alpha, weights, deltas):
        delta = np.matmul(weights.T, deltas ) * alpha * (1 - alpha)
        delta = delta[1:]
        return delta

    def delta_output(self, alpha_output, real_output):
        # print("real output", real_output)
        # print("alpha output", alpha_output.T)
        delta = alpha_output.T - real_output
        delta = delta.T
        # print("delta_output", delta)
        return delta



    def update_weights(self, gradients):
        self.weights -= (self.learning_rate * gradients)

    def transform_to_class(self, alphas):
        output = np.zeros(len(alphas))
        output[np.argmax(alphas)] = 1
        return output

    def function_j(self, output):
        return np.sum(output)
