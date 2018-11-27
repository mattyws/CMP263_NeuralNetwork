import csv
import os
import pickle

import pandas as pd
import random
import kfold
import collections

import helper



#WINE
from neuralnetwork import NeuralNetwork

data = pd.read_csv("resources/Wine/wine.data.txt", header=None, sep=',')
# Cria colunas
data.columns = ["TipoVinho", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
                     "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
                     "OD280/OD315 of diluted wines", "Proline"]


labels = data[data.columns[0]]
data = data.drop(data[data.columns[0:1]], axis=1)

#IONOSPHERE
# data = pd.read_csv("resources/Ionosphere/ionosphere.data.txt", header=None, sep=',', prefix='H')
# labels = data.drop(columns=data.columns[:-1])
# data = data.drop(columns=data.columns[-1])

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


# semente para o número aleatório
nseed = 100
random.seed(nseed)

directory_results = "nn_results/"
min_number_trees = 3
max_number_trees = 21
with open(directory_results+"/result.csv", 'w') as csv_result_file:
    writer = csv.writer(csv_result_file, delimiter=',')
    writer.writerow(['Precision', 'Recall', 'F1_Score'])
    k = 10
    kf = kfold.KFold()
    kf.make_kfold(data, labels, k, nseed)
    folds_precision = []
    folds_recall = []
    folds_f1 = []

    for i in range(k):
        print("======= Fold {} ======".format(i))
        fold_directory = directory_results+str(i)
        if not os.path.exists(fold_directory):
            os.makedirs(fold_directory)

        data_test, labels_test, data_train, labels_train = kf.get_data_test_train()
        kf.update_test_index()
        nn = NeuralNetwork(layers=[2], lamb=.25, max_iter=1000, learning_rate=0.0001,
                           threshold=0.00001)
        total_labels = len(labels_train[labels_train.columns[0]].unique())
        mean = data_train.mean()
        std = data_train.std()
        data_train = helper.normalize(data_train, mean, std)
        labels_train = helper.one_hot_encode(labels_train[labels_train.columns[0]])
        nn.train(data_train, labels_train, total_labels)
        exit(0)
        data_test = helper.normalize(data_test, mean, std)
        predictions = helper.one_hot_decode(nn.predict(data_test))
        macro_p = helper.macro_precision(labels_test[labels_test.columns[0]].tolist(), predictions)
        macro_r = helper.macro_recall(labels_test[labels_test.columns[0]].tolist(), predictions)
        macro_f1 = helper.macro_f1_score(labels_test[labels_test.columns[0]].tolist(), predictions)
        folds_precision.append(macro_p)
        folds_recall.append(macro_r)
        folds_f1.append(macro_f1)

        results_text = ""
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

        with open(fold_directory+'/forest.pkl', 'wb') as output:
            pickle.dump(nn, output, pickle.HIGHEST_PROTOCOL)

        print(results_text)

    with open(directory_results+"/result", "w") as result_file:
        result_file.write("Folds precision: {}\n".format(folds_precision))
        result_file.write("Folds recall: {}\n".format(folds_recall))
        result_file.write("Folds f1 score: {}\n".format(folds_f1))
        result_file.write("Folds precision average: {0:.2f}%\n".format( (sum(folds_precision) / len(folds_precision)) *100 ))
        result_file.write("Folds recall average: {0:.2f}%\n".format( (sum(folds_recall) / len(folds_recall)) * 100 ))
        result_file.write("Folds f1 score average: {0:.2f}%\n".format( (sum(folds_f1) / len(folds_f1)) *100 ))
    writer.writerow(["{0:.2f}".format((sum(folds_precision) / len(folds_precision)) *100),
                     "{0:.2f}".format((sum(folds_recall) / len(folds_recall)) * 100),
                     "{0:.2f}".format((sum(folds_f1) / len(folds_f1)) *100)])