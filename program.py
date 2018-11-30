import csv
import os
import pickle
from time import time

import pandas as pd
import random
import kfold
import collections

import helper
from sklearn.model_selection import ParameterGrid



#WINE
from neuralnetwork import NeuralNetwork

data = pd.read_csv("resources/Wine/wine.data.txt", header=None, sep=',')
# Cria colunas
data.columns = ["TipoVinho", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
                     "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
                     "OD280/OD315 of diluted wines", "Proline"]
dataset_directory = "resources/Wine/"

labels = data[data.columns[0]]
data = data.drop(data[data.columns[0:1]], axis=1)

#IONOSPHERE
# data = pd.read_csv("resources/Ionosphere/ionosphere.data.txt", header=None, sep=',', prefix='H')
# labels = data[data.columns[-1]]
# labels = helper.discretize(labels)
# data = data.drop(columns=data.columns[-1])
# data = data.drop(columns="H1") # TODO: Columns with only 0, remove
# dataset_directory = "resources/Ionosphere/"

#BrestCancer
# data = pd.read_csv("resources/BreastCancer/breast-cancer-wisconsin.data.txt", header=None, sep=',', prefix='H')
# data = data[data['H6'] != '?']
# data['H6'] = data['H6'].apply(pd.to_numeric)
# data = data.dropna()
# data = data.reset_index()
# labels = data.drop(columns=data.columns[:-1])
# data = data.drop(columns=data.columns[-1])
# #Drop first column because they are ID's
# data = data.drop(columns=data.columns[0])
# dataset_directory = "resources/BreastCancer/"

#PIMA
# data = pd.read_csv("resources/Pima/pima.tsv", sep='\t')
# labels = data.drop(columns=data.columns[:-1])
# data = data.drop(columns=data.columns[-1])
# dataset_directory = "resources/Pima/"

parameter_grid = {
    'layers': [[70],[50,20], [100,100],[200]],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'max_iter': [200, 500, 700]
}

parameter_grid = ParameterGrid(parameter_grid)
# semente para o número aleatório
nseed = 100
random.seed(nseed)

with open(dataset_directory + "nn_results/results_final.csv", 'w') as final_results_handler:
    final_results = csv.writer(final_results_handler, delimiter=';')
    final_results.writerow(['Test_num', 'Parameters', 'Precision', 'Recall', 'F1_Score'])
    for test_num, parameter in enumerate(parameter_grid):
        directory_results = dataset_directory + "nn_results/test_"+str(test_num)
        time_total = 0
        if not os.path.exists(directory_results):
            os.makedirs(directory_results)
        k = 10
        kf = kfold.KFold()
        kf.make_kfold(data, labels, k, nseed)
        folds_precision = []
        folds_recall = []
        folds_f1 = []

        for i in range(k):
            print("======= Fold {} ======".format(i))
            fold_directory = directory_results+'/fold_{}'.format(i)
            if not os.path.exists(fold_directory):
                os.makedirs(fold_directory)

            data_test, labels_test, data_train, labels_train = kf.get_data_test_train()
            kf.update_test_index()
            nn = NeuralNetwork(layers=parameter["layers"], lamb=.25, max_iter=parameter["max_iter"],
                               learning_rate=parameter["learning_rate"], threshold=1e-6)
            total_labels = len(labels_train[labels_train.columns[0]].unique())
            mean = data_train.mean()
            std = data_train.std()
            data_train = helper.normalize(data_train, mean, std)
            labels_train, encode_dict = helper.one_hot_encode(labels_train[labels_train.columns[0]])

            begin = time()
            nn.train(data_train, labels_train, total_labels)
            end = time()

            data_test = helper.normalize(data_test, mean, std)
            predictions = helper.one_hot_decode(nn.predict(data_test), encode_dict)

            macro_p = helper.macro_precision(labels_test[labels_test.columns[0]].tolist(), predictions)
            macro_r = helper.macro_recall(labels_test[labels_test.columns[0]].tolist(), predictions)
            macro_f1 = helper.macro_f1_score(labels_test[labels_test.columns[0]].tolist(), predictions)
            folds_precision.append(macro_p)
            folds_recall.append(macro_r)
            folds_f1.append(macro_f1)

            time_total += (end-begin)/60

            results_text = ""
            results_text += "Time took to train: {} minutes\n".format((end-begin)/60)
            results_text += "Number of iterations {}\n".format(nn.iter)
            results_text += "Ground truth: {}\n".format(labels_test[labels_test.columns[0]].tolist())
            results_text += "Predictions: {}\n".format(predictions)
            results_text += "Precision per class: {}\n".format(helper.precision(labels_test[labels_test.columns[0]].tolist(), predictions))
            results_text += "Recall per class: {}\n".format(helper.recall(labels_test[labels_test.columns[0]].tolist(), predictions))
            results_text += "F1 Scrore per class: {}\n".format(helper.f1_score(labels_test[labels_test.columns[0]].tolist(), predictions))
            results_text += "Macro precision: {0:.2f}%\n".format(macro_p * 100)
            results_text += "Macro recall: {0:.2f}%\n".format(macro_r * 100)
            results_text += "Macro F1 Score: {0:.2f}%\n".format(macro_f1 * 100)

            with open(fold_directory+"/result", "w") as text_file:
                text_file.write(results_text)

            with open(fold_directory+'/nn.pkl', 'wb') as output:
                pickle.dump(nn, output, pickle.HIGHEST_PROTOCOL)

            print(results_text)

        with open(directory_results+"/result", "w") as result_file:
            result_file.write("Parameter used to train {}\n".format(parameter))
            result_file.write("Total time to cross validate {}\n".format(time_total))
            result_file.write("Folds precision: {}\n".format(folds_precision))
            result_file.write("Folds recall: {}\n".format(folds_recall))
            result_file.write("Folds f1 score: {}\n".format(folds_f1))
            result_file.write("Folds precision average: {0:.2f}%\n".format( (sum(folds_precision) / len(folds_precision)) *100 ))
            result_file.write("Folds recall average: {0:.2f}%\n".format( (sum(folds_recall) / len(folds_recall)) * 100 ))
            result_file.write("Folds f1 score average: {0:.2f}%\n".format( (sum(folds_f1) / len(folds_f1)) *100 ))
        final_results.writerow([test_num, parameter,
                        "{0:.2f}".format((sum(folds_precision) / len(folds_precision)) *100),
                        "{0:.2f}".format((sum(folds_recall) / len(folds_recall)) * 100),
                        "{0:.2f}".format((sum(folds_f1) / len(folds_f1)) *100)
                                ])