import pandas as pd
import numpy as np

class KFold:

    data_kfold = []
    labels_kfold = []

    indice_teste = 0


    def make_kfold(self, data, labels, k, seed):
        """
        Cria k folds
        :param data: a base de dados contendo a coluna sem labels
        :param labels: labels dos dados
        :param k: quantidade de folds
        :param seed: semente para aleatorização
        """

        np.random.seed(seed)

        data = pd.concat([data, labels], axis=1)

        #print("Imprimindo dados com labels dentro da função make_kfold")
        #print(data.head(2))

        data = data.iloc[np.random.permutation(len(data))]

        labels = data.iloc[:, len(data.columns)-1]
        #print("Imprimindo labels após aleatoridade em make_kfold")
        #print(labels.head(2))

        grouped = data.groupby(labels)

        new_df = pd.DataFrame()
        for key, item in grouped:
            g = grouped.get_group(key)
            new_df = new_df.append(g, ignore_index=True)

        #labels = new_df.iloc[:, len(new_df.columns)-1]
        #new_df = new_df.drop(new_df[new_df.columns[len(new_df.columns)-1:len(new_df.columns)]], axis=1)
        #print("Imprimindo labels agrupados")
        #print(labels.head(2))

        #print("Imprimindo new_df apos agrupamento")
        #print(new_df.head(2))

        df_array = []
        df_labels = []
        for j in range(k):
            df_array.append(pd.DataFrame())
            df_labels.append(pd.DataFrame())

        j = 0
        for i in range(len(new_df)):
            if j >= k:
                j = 0
            df_array[j] = df_array[j].append(new_df.iloc[[i]], ignore_index=True)
            j += 1

        #print("Imprimindo posição em 1 df_arrays")
        #print(df_array[1].head(2))

        #print("Imprimindo df_labels")
        #print(df_array[1].iloc[:, len(df_array[1].columns)-1])
        j = 0
        for i in range(k):
            if j >= k:
                j = 0
            df_labels[j] = pd.concat([df_labels[j], df_array[j][df_array[j].columns[len(df_array[j].columns) - 1:len(df_array[j].columns)]]], axis=1)
            df_array[j] = df_array[j].drop(df_array[j][df_array[j].columns[len(df_array[j].columns) - 1:len(df_array[j].columns)]], axis=1)
            j += 1

        #print("labels separados:")
        #print(df_labels[1])

        self.data_kfold = df_array
        self.labels_kfold = df_labels
        self.indice_teste = 0


    def get_data_test_train(self):
        """
        Retorna para o usuário quais são os dados de testes e de treinamento atuais
        :return:: um DataFrame de teste, os labels de teste, um DataFrame com os dados de treinamento e os labels de treino
        """

        data_test = self.data_kfold[self.indice_teste]
        labels_test = self.labels_kfold[self.indice_teste]

        data_train = pd.DataFrame()
        labels_train = pd.DataFrame()

        for j in range(len(self.data_kfold)):
            if j != self.indice_teste:
                data_train = pd.concat([data_train, self.data_kfold[j]])
                labels_train = pd.concat([labels_train, self.labels_kfold[j]])

        return data_test, labels_test, data_train, labels_train


    def update_test_index(self):
        self.indice_teste += 1
