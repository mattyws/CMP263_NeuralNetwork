import csv
import numpy as np
import json
import matplotlib.pyplot as plt

resource_paths = {"BreastCancer": "resources/BreastCancer/nn_results/",
                  "Wine": "resources/Wine/nn_results/",
                  "Pima": "resources/Pima/nn_results/",
                  "Ionosphere": "resources/Ionosphere/nn_results/"}
top_n = 5
for dataset, resource_files in resource_paths.items():
    with open(resource_files+'results_final.csv', 'r') as csv_result_file:
        print(dataset)
        spamreader = csv.DictReader(csv_result_file, delimiter=';')
        f1_scores = []
        precision_scores = []
        recall_scores = []
        parameter = []
        tests_num = []
        for row in spamreader:
            tests_num.append(int(row['Test_num']))
            precision_scores.append(float(row['Precision']))
            recall_scores.append(float(row['Recall']))
            f1_scores.append(float(row['F1_Score']))
            parameter.append(json.loads(row['Parameters'].replace('\'', '\"')))
        tests_num = np.array(tests_num)
        f1_scores = np.array(f1_scores)
        precision_scores = np.array(precision_scores)
        recall_scores = np.array(recall_scores)
        parameter = np.array(parameter)

        # Get the higher 5 values indexes from the f1 vector
        f1_higher_index = f1_scores.argsort()[-top_n:]
        print(dataset, f1_scores[f1_higher_index], parameter[f1_higher_index])

        # Create the deterioration fbased on learning rate for the best configuration for the dataset
        best_param = parameter[f1_higher_index[-1]]
        # Get index with the same layer config
        parameters_indexes = []
        for i in range(len(parameter)):
            if parameter[i]["layers"] == best_param["layers"]:
                parameters_indexes.append(i)

        # Get the indexes based on ordered by learning rate
        ordered_by_learning_rate = sorted(parameter[parameters_indexes], key=lambda k: k['learning_rate'])
        parameters_indexes = []
        for p in ordered_by_learning_rate:
            for i in range(len(parameter)):
                if parameter[i]['max_iter'] == p['max_iter'] and parameter[i]['learning_rate'] == p['learning_rate']\
                        and parameter[i]['layers'] == p['layers']:
                    parameters_indexes.append(i)
                    break
        parameters_indexes = np.array(parameters_indexes)

        plt.xlabel("Taxa de Aprendizagem")
        plt.ylabel("F1 Score")
        plt.axis([1e-5, 1e-3, min(f1_scores) - 5, 105])
        plt.xticks([1e-3, 1e-4, 1e-5], ('1e-3', '1e-4', '1e-5'))
        for i in range(int(len(parameters_indexes)/3)):
            learning_rates = []
            max_iter = 0
            for p in parameter[parameters_indexes[[i, i+3, i+6]]]:
                learning_rates.append(p["learning_rate"])
                max_iter = p["max_iter"]
            plt.plot( learning_rates, f1_scores[parameters_indexes[[i, i+3, i+6]]], label=str(max_iter))
        plt.legend(loc="lower right")
        plt.title(dataset + " - Impacto da taxa de aprendizagem entre iterações")
        plt.savefig(resource_files + "f1_learning_rate.png", format="png")
        plt.clf()

        # Comparing distribution of neurons between layers
        parameters_indexes = []
        for i in range(len(parameter)):
            if sum(parameter[i]["layers"]) == sum(best_param["layers"]) and parameter[i]["max_iter"] == best_param["max_iter"]:
                parameters_indexes.append(i)

        plt.xlabel("Taxa de Aprendizagem")
        plt.ylabel("F1 Score")
        plt.axis([1e-5, 1e-3, min(f1_scores) - 5, 105])
        plt.xticks([1e-3, 1e-4, 1e-5], ('1e-3', '1e-4', '1e-5'))
        learning_rates = []
        layers = []
        for p in parameter[parameters_indexes[0:3]]:
            learning_rates.append(p["learning_rate"])
            layers = p['layers']
        plt.plot( learning_rates, f1_scores[parameters_indexes[0:3]], label=str(layers) )
        learning_rates = []
        for p in parameter[parameters_indexes[3:]]:
            learning_rates.append(p["learning_rate"])
            layers = p['layers']
        plt.plot(learning_rates, f1_scores[parameters_indexes[3:]], label=str(layers))
        plt.legend(loc="lower right")
        plt.title(dataset+" - Diferença entre distribuição de neurônios")
        plt.savefig(resource_files + "f1_layers.png", format="png")
        plt.clf()

        # Plot f1 for each fold for the best results
        best_tests_num = tests_num[f1_higher_index]
        plt.xlabel("Folds")
        plt.ylabel("F1 Score")
        plt.axis([1, 10, min(f1_scores) - 5, 105])
        plt.xticks(range(1, 11), labels=range(1, 11))
        for num in best_tests_num:
            with open(resource_files+"test_{}/result".format(num)) as test_result_file_handler:
                for line in test_result_file_handler:
                    if "f1 score" in line:
                        test_f1_scores = line.split(": ")[1].strip().replace('[', '').replace(']', '').split(', ')
                        test_f1_scores = np.array(test_f1_scores, dtype=float) * 100
                        print(test_f1_scores)
                        plt.plot([1,2,3,4,5,6,7,8,9,10], test_f1_scores, label=parameter[num])
                        break
        if dataset == "Pima":
            plt.legend(loc="upper right")
        else:
            plt.legend(loc="lower right")
        plt.title(dataset + " - F1 por fold para as melhores configurações de rede")
        plt.savefig(resource_files + "f1_per_fold.png", format="png")
        plt.clf()

            # parameters_neurons = dict()
        # for p in parameter:
        #     sum_layer = sum(p['layers'])
        #     if sum_layer not in parameters_neurons.keys():
        #         parameters_neurons[sum_layer] = dict()
        #     if p["max_iter"] not in parameters_neurons[sum_layer].keys():
        #         parameters_neurons[sum_layer][p["max_iter"]] = dict()
        #     if p["learning_rate"] not in parameters_neurons[sum_layer][p["max_iter"]].keys():
        #         parameters_neurons[sum_layer][p["max_iter"]][p["learning_rate"]] = []
        #     parameters_neurons[sum_layer][p["max_iter"]][p["learning_rate"]].append(p)
        # print(parameters_neurons)

        # plt.xlabel("Número de Árvores")
        # plt.ylabel("F1 Score")
        # plt.plot(num_trees, f1_scores)
        # plt.axis([min(num_trees), max(num_trees), min(f1_scores)-5, 100])
        # plt.xticks(num_trees)
        # plt.savefig(resource_files+"f1_score_grow.png", format="png")
        # plt.clf()
        #
        # plt.xlabel("Número de Árvores")
        # plt.ylabel("Precisão")
        # plt.plot(num_trees, precision_scores)
        # plt.axis([min(num_trees), max(num_trees), min(precision_scores) - 5, 100])
        # plt.xticks(num_trees)
        # plt.savefig(resource_files + "precision_grow.png", format="png")
        # plt.clf()
        #
        # plt.xlabel("Número de Árvores")
        # plt.ylabel("Recall")
        # plt.plot(num_trees, recall_scores)
        # plt.axis([min(num_trees), max(num_trees), min(recall_scores) - 5, 100])
        # plt.xticks(num_trees)
        # plt.savefig(resource_files + "recall_grow.png", format="png")
        # plt.clf()