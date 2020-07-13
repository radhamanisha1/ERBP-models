import torch
import numpy as np
import itertools
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F
from pytorchtools import EarlyStopping
import torch.utils.data as utils_data
from sklearn.metrics import accuracy_score
from utils import utils_func

parser = argparse.ArgumentParser(description='RBP late fusion model')
parser.add_argument('--model', type=str, default='rnn', help='type of recurrent net (rnn, gru, lstm)')
parser.add_argument('--save', type=str,  default='rnn-model.pt',help='path to save the final model')
parser.add_argument('--bptt', type=int, default= 8,help='sequence length')
parser.add_argument('--hidden_size', type=int, default= 8,help='hidden size')
args = parser.parse_args()

file = open('late-fusion.txt','w')

def create_datasets(batch_size,seq_len,dataset):
    # percentage of training set to use as validation
    valid_size = 0.2
    data = [line.strip() for line in open(dataset, 'r')]
    # data = data[:500]
    ntokens = len(set(data))
    train_data, test_data, train_d, test_d, training_data, testing_data = [], [], [], [], [], []
    kf = KFold(n_splits=10)
    # data = data[:50]
    for train, test in kf.split(data):
        train_data.append(train)
        test_data.append(test)

    train_d = list(itertools.chain.from_iterable(train_data))
    test_d = list(itertools.chain.from_iterable(test_data))

    for i in train_d:
        training_data.append(data[i])

    for i in test_d:
        testing_data.append(data[i])

    vocab_size = list(set(data))
    char_to_int = dict((c, i) for i, c in enumerate(vocab_size))

    len_train_data = len(training_data)
    m_index = int(len_train_data / 10)
    k_index = len_train_data - m_index
    train_data = training_data[:k_index]
    validation_data = training_data[k_index:]

    training_data = [char_to_int[char] for char in train_data]
    validation_data = [char_to_int[char] for char in validation_data]
    testing_data = [char_to_int[char] for char in testing_data]

    seq_len = seq_len + 1

    data_modified1 = zip(*[iter(training_data)] * seq_len)
    data_modified2 = zip(*[iter(validation_data)] * seq_len)
    data_modified3 = zip(*[iter(testing_data)] * seq_len)
    # convert into list
    data_modified_list1 = [list(i) for i in data_modified1]
    data_modified_list2 = [list(i) for i in data_modified2]
    data_modified_list3 = [list(i) for i in data_modified3]

    train_inputs, train_targets = [], []
    valid_inputs, valid_targets = [], []
    test_inputs, test_targets = [], []

    for i in data_modified_list1:
        train_inputs.append(i[:-1])
        train_targets.append(i[-1:])

    for i in data_modified_list2:
        valid_inputs.append(i[:-1])
        valid_targets.append(i[-1:])

    for i in data_modified_list3:
        test_inputs.append(i[:-1])
        test_targets.append(i[-1:])


    train_utils = utils_func(data_modified_list1, seq_len)
    valid_utils = utils_func(data_modified_list2, seq_len)
    test_utils = utils_func(data_modified_list3, seq_len)

    valid_tokens = set(data)
    # print(test_inputs[:1])
    # print(dr_data1[:1])
    print(len(vocab_size))
    dr_data1 = [torch.FloatTensor(i) for i in train_utils]
    dr_data2 = [torch.FloatTensor(i) for i in valid_utils]
    dr_data3 = [torch.FloatTensor(i) for i in test_utils]

    new_dr1 = [0] * len(vocab_size)
    new_dr2 = [0] * len(vocab_size)
    new_dr3 = [0] * len(vocab_size)

    # print(new_dr1)
    # for k in test_inputs:
    #     for p in k:
    #         for i in new_dr:
    #             for j in dr_data1:
    #                 i[p] = j

    flat_list1 = [item for sublist in train_inputs for item in sublist]
    flat_list2 = [item for sublist in dr_data1 for item in sublist]

    flat_list3 = [item for sublist in valid_inputs for item in sublist]
    flat_list4 = [item for sublist in dr_data2 for item in sublist]

    flat_list5 = [item for sublist in test_inputs for item in sublist]
    flat_list6 = [item for sublist in dr_data3 for item in sublist]

    gf1 = list(zip(flat_list1, flat_list2))
    gf2 = list(zip(flat_list3, flat_list4))
    gf3 = list(zip(flat_list5, flat_list6))

    data_modified1 = list(zip(*[iter(gf1)] * seq_len))
    data_modified2 = list(zip(*[iter(gf2)] * seq_len))
    data_modified3 = list(zip(*[iter(gf3)] * seq_len))

    # print('gf', gf[1])
    bp1,bp2,bp3 = [],[],[]

    for k in data_modified1:
        new_dr1 = [0] * len(vocab_size)
        for g in k:
            # print('g', g)
            new_dr1[g[0]] = g[1]
        bp1.append(new_dr1)

    for k in data_modified2:
        new_dr2 = [0] * len(vocab_size)
        for g in k:
            # print('g', g)
            new_dr2[g[0]] = g[1]
        bp2.append(new_dr2)

    for k in data_modified3:
        new_dr3 = [0] * len(vocab_size)
        for g in k:
            # print('g', g)
            new_dr3[g[0]] = g[1]
        bp3.append(new_dr3)
    # # for i in new_dr:
    # for k in data_modified[:1]:
    #     new_dr[k[0]] = k[1]
    #     bp.append(new_dr)

    # print('len og fip', len(bp))

    dr_data1 = [torch.FloatTensor(i) for i in bp1]
    dr_data2 = [torch.FloatTensor(i) for i in bp2]
    dr_data3 = [torch.FloatTensor(i) for i in bp3]


    t1 = [torch.LongTensor(np.array(i)) for i in train_inputs]
    #t1 = [torch.LongTensor(np.array(i)) for i in DR_train_list]
    f1 = [torch.LongTensor(np.array(i)) for i in train_targets]
    # print(t1[:1])
    #t2 = [torch.LongTensor(np.array(i)) for i in DR_valid_list]
    t2 = [torch.LongTensor(np.array(i)) for i in valid_inputs]
    f2 = [torch.LongTensor(np.array(i)) for i in valid_targets]

    #t3 = [torch.LongTensor(np.array(i)) for i in DR_test_list]
    t3 = [torch.LongTensor(np.array(i)) for i in test_inputs]
    f3 = [torch.LongTensor(np.array(i)) for i in test_targets]

    training_samples = utils_data.TensorDataset(torch.stack(t1), torch.stack(f1))
    validation_samples = utils_data.TensorDataset(torch.stack(t2), torch.stack(f2))
    testing_samples = utils_data.TensorDataset(torch.stack(t3), torch.stack(f3))

    print('no of training samples,', len(training_samples))
    print('no of validation samples', len(validation_samples))
    print('no of testing samples', len(testing_samples))

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(training_samples,
                                               batch_size=batch_size,
                                               num_workers=0)

    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(validation_samples,
                                               batch_size=batch_size,
                                               num_workers=0)

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(testing_samples,
                                              batch_size=batch_size,
                                              num_workers=0)

    return train_loader, test_loader, valid_loader, ntokens, dr_data1,dr_data2,dr_data3


class RNNModule(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, model, n_layers):
        super(RNNModule, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.i2h = torch.nn.Linear(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = torch.nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = torch.nn.LSTM(hidden_size, hidden_size, n_layers)
        elif self.model == "rnn":
            self.rnn = torch.nn.RNN(hidden_size, hidden_size, n_layers, nonlinearity='relu')
        self.h2o = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden,dr_data):
        batch_size = input.size(0)
        encoded = self.i2h(input.view(1,-1))
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        out = self.h2o(output.view(batch_size, -1))
        out1 = F.softmax(out)
        output = torch.add(out1,dr_data)
        return output, hidden


    def init_hidden(self, batch_size):

        if self.model == "lstm":
            return (torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

        return torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


def train_model(model, batch_size, patience, n_epochs,dr_data1, dr_data2):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for l in dr_data1:
            hidden = model.init_hidden(batch_size)
            for batch, (data, target) in enumerate(train_loader):
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # print('data', data)
                # print('tar', target)
                # forward pass: compute predicted outputs by passing inputs to the model
                # output, hidden = model(Variable(data).float(), hidden)
                output, hidden = model(Variable(data).float(), hidden,Variable(l).view(-1,ntokens))
                # calculate the loss
                loss = criterion(output, Variable(target).view(-1))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward(retain_graph=True)
                # perform a single optimization step (parameter update)
                optimizer.step()
                # record training loss
                train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        hidden = model.init_hidden(batch_size)
        for k in dr_data2:
            for data, target in valid_loader:
                # forward pass: compute predicted outputs by passing inputs to the model
                output, hidden = model(Variable(data).float(), hidden,Variable(k).view(-1,ntokens))
                # calculate the loss
                loss = criterion(output, Variable(target).view(-1))
                # record validation loss
                valid_losses.append(loss.item())

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)
            file.write(print_msg)
            file.write('***************')
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses



batch_size = 1
n_epochs = 10

dataset = ['data/pitch-plots/data_kinder.txt', '/data/pitch-plots/data_jogoslav.txt', '/data/pitch-plots/data_nova-scotia.txt', '/data/pitch-plots/data_schweiz.txt',  '/data/pitch-plots/data_oesterrh.txt', '/data/pitch-plots/data_bach.txt', '/data/pitch-plots/data_shanxi.txt', '/data/pitch-plots/data_elsass.txt']
bptt = [5,10,20,30,50]
#bptt = [2]
rnn_type = ["rnn", "lstm", "gru"]
for i in bptt:
    for d in dataset:
        train_loader, test_loader, valid_loader, ntokens, dr_data1, dr_data2, dr_data3 = create_datasets(batch_size,i,d)
        # print('star',(len(diff1[:0]) + i)-1)
        for k in rnn_type:
            model = RNNModule(i,i, ntokens, k, 1)
            file.write('len : %d, model_type : %s \n' % (i,k) )
            file.write('*******************')

            # specify loss function
            criterion = nn.CrossEntropyLoss()

            # specify optimizer
            optimizer = torch.optim.Adam(model.parameters())

            # early stopping patience; how long to wait after last time validation loss improved.
            patience = 20

            model1, train_loss, valid_loss = train_model(model, batch_size, patience, n_epochs,dr_data1,dr_data2)

            test_loss = 0.0
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            correct,total = 0,0
            model1.eval() # prep model for evaluation
            for j in dr_data3:
                hidden = model.init_hidden(batch_size)
                for data, target in test_loader:
                    if len(target.data) != batch_size:
                        break
                # forward pass: compute predicted outputs by passing inputs to the model
                    output, hidden = model1(Variable(data).float(), hidden, Variable(j).view(-1,ntokens))
                # calculate the loss
                    loss = criterion(output, Variable(target).view(-1))
                # record validation loss
                # update test loss
                    test_loss += loss.item()*data.size(0)
                # convert output probabilities to predicted class
                    _, pred = torch.max(output, 1)
                    values, tar = torch.max(output, 1)

                    total += target.size(0)
                    correct += (tar.view(-1, 1) == Variable(target)).sum().item()

            # calculate and print avg test loss
                test_loss = test_loss/len(test_loader.dataset)
                print('Test Loss: {:.6f}\n'.format(test_loss))

                # print('Test Accuracy of the model : {} %'.format(100 * correct / total))
                # acc = (100*correct/ total)

                file.write('test_loss : %d, acc : %d \n' % (test_loss,acc))
                file.write('*******end**************')

file.close()


