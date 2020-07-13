from scipy.io import loadmat
import torch
import numpy as np


def data_generator(dataset):
    if dataset == "JSB":
        print('loading JSB data...')
        data = loadmat('./mdata/JSB_Chorales.mat')
    elif dataset == "Nott":
        print('loading Nott data...')
        data = loadmat('./mdata/Nottingham.mat')

    print(data['traindata'][0][0])
    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]

    # print(X_train)
    for data in [X_train, X_valid, X_test]:
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float32))

    return X_train, X_valid, X_test, data


X_train, X_valid, X_test, data = data_generator('JSB')
# print(data)
