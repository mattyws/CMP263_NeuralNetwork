import collections

import numpy as np
import pandas as pd

def discretize(labels):
    total_classe = len(labels.unique())
    uniques = labels.unique()
    new_labels = []
    for label in labels:
        for i in range(total_classe):
            if label == uniques[i]:
                new_labels.append(i)
                break
    return pd.Series(new_labels)

def normalize(data, mean, std):
    data = (data - mean) / std
    return data

def one_hot_decode(labels, encode_dict):
    new_labels = []
    for d in labels:
        # print(encode_dict)
        for key, value in encode_dict.items():
            if np.argmax(value) == np.argmax(d):
                new_labels.append(key)
    return new_labels

def one_hot_encode(labels):
    total_classe = len(labels.unique())
    new_dict = dict()
    for i in range(0, total_classe):
        new_dict[labels.unique()[i]] = np.zeros(total_classe)
        new_dict[labels.unique()[i]][i] = 1
    # print(new_dict)
    new_labels = []
    for i in labels:
        new_labels.append(new_dict[i])
    new_labels = pd.Series(new_labels)
    return new_labels, new_dict

def recall(true, predicted):
    hit = [x for x, y in zip(true, predicted) if x == y]
    true_values_count = collections.Counter(true)
    hit_count = collections.Counter(hit)
    recalls = {}
    for key in true_values_count.keys():
        if key in hit_count.keys():
            recalls[key] = hit_count[key] / true_values_count[key]
        else:
            recalls[key] = 0
    return recalls

def precision(true, predicted):
    hit = [x for x, y in zip(true, predicted) if x == y]
    predicted_values_count = collections.Counter(predicted)
    hit_count = collections.Counter(hit)
    precisions = {}
    for key in predicted_values_count.keys():
        if key in hit_count.keys():
            precisions[key] = hit_count[key] / predicted_values_count[key]
        else :
            precisions[key] = 0
    return precisions

def macro_recall(true, predicted):
    recalls = recall(true, predicted)
    macro_recall = 0
    for key in recalls.keys():
        macro_recall += recalls[key]
    macro_recall /= len(recalls.keys())
    return macro_recall

def macro_precision(true, predicted):
    precisions = precision(true, predicted)
    macro_precision = 0
    for key in precisions.keys():
        macro_precision += precisions[key]
    macro_precision /= len(precisions.keys())
    return macro_precision

def f1_score(true, predicted):
    precisions = precision(true, predicted)
    recalls = recall(true, predicted)
    f1_scores = {}
    for key in precisions.keys():
        if precisions[key] == 0 and recalls[key] == 0:
            f1_scores[key] = 0
        else:
            f1_scores[key] = 2 * ( (precisions[key]*recalls[key]) / (precisions[key]) + recalls[key] )
    return f1_scores

def macro_f1_score(true, predicted):
    macro_p = macro_precision(true, predicted)
    macro_r = macro_recall(true, predicted)
    if macro_p == 0 and macro_r == 0:
        return 0
    return 2 * ((macro_p * macro_r)/(macro_p + macro_r))