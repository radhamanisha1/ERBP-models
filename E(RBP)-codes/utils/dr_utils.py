import itertools
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
import numpy as np

def utils_func(data_modified_list, seq_len):
    inputs, targets = [],[]

    for i in data_modified_list:
        inputs.append(i[:-1])
        targets.append(i[-1:])

    #convert into tensors
    two_comb = []

    for i in inputs:
        two_comb.extend(list(itertools.combinations(i,2)))

    two_comb_list = [list(i) for i in two_comb]
    diff = [abs(x-y) for x,y in two_comb]

    #get binary differences
    # diff_binary = []
    #
    # for i in diff:
    #     if i==0:
    #         diff_binary.append(i)
    #     elif i!=0:
    #         i=1
    #         diff_binary.append(i)

    model = MLPRegressor(activation='logistic')

    # two_comb_list = [list(i) for i in two_comb_list]
    print(two_comb_list[:1])
    print(diff[:1])
    model.fit(two_comb_list,diff)

    count_zero, count_one = 0,0

    for i in diff:
        if i==0:
            count_zero+=1
        elif i==1:
            count_one+=1

    #try to predict on test data
    test_inputs = []
    #repplicate target tensor to make it compatible with zip!!!
    test1 = [[i]*seq_len for i in targets]

    #Flat list such that each list of list merges into a single list..
    flat_list = list(list(itertools.chain.from_iterable(i)) for i in test1)

    for i,j in zip(targets,flat_list):
        test_inputs.append(zip(i,j))

    #modify these test inputs-- tuples into lists
    #flatten them again
    cc = [item for sublist in test_inputs for item in sublist]
    test_inputs_list = [list(c) for c in cc]
    #NEW LINE ADDED
    diff1 = [abs(x-y) for x,y in cc]
    print('len of diff1', len(diff1))
    lk = len(diff1[:0])
    # np.corrcoef(diff,diff1) #for bach

    k = model.predict(test_inputs_list)
    #print(type(k))
    #coefficients = [coef.shape for coef in model.coefs_]
    #print(coefficients)
    bin_diff_chunks = zip(*[iter(diff)]*lk)
    bin_out_chunks = zip(*[iter(k)]*seq_len)

    print(type(bin_out_chunks))
    bin_out_chunks = [list(i) for i in bin_out_chunks]
    print(normalize(bin_out_chunks))
    final_chunks_dict = dict(zip(bin_diff_chunks,bin_out_chunks))

    file = ''
    np.save('test.txt', bin_out_chunks)
    f = open('test.txt','w')
    for k,v in final_chunks_dict.items():
        print>>f,k,v

    b = np.load('test.txt.npy')

    return b
