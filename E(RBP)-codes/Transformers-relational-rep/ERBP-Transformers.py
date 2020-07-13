import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from sklearn.model_selection import GridSearchCV,KFold
import erbp_model as model1

parser = argparse.ArgumentParser(description='PyTorch Music Language Model')
parser.add_argument('--model', type=str, default='Transformer',
                    help='type of recurrent net (LSTM,Transformer)')
parser.add_argument('--emsize', type=int, default=20,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=20,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.02,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=10,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='erbp.pt',
                    help='path to save the final model')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")
kf = KFold(n_splits=10)

def data_generator(dataset_type):
    train_data, test_data, train_d, test_d, training_data, testing_data, validation_data = [], [], [], [], [], [], []
    url = '/data/pitch-plots/' + dataset_type + '.txt'
    data = [line.strip() for line in open(url, 'r')]
    for train, test in kf.split(data):
        train_data.append(train)
        test_data.append(test)
    len_train_data = len(train_data)
    print(len_train_data)
    m_index = math.floor(len_train_data / 10)

    k_index = len_train_data - m_index
    train_data1 = train_data[:k_index]
    valid_data = train_data[k_index:]

    for i in train_data1:
        for k in i:
            training_data.append(data[k])

    # print(training_data)
    for i in valid_data:
        for k in i:
            validation_data.append(data[k])

    for i in test_data:
        for k in i:
            testing_data.append(data[k])

    vocab_size = list(set(data))
    # print(vocab_size)
    print('len of vocab', len(vocab_size))
    char_to_int = dict((c, i) for i, c in enumerate(vocab_size))

    training_data = [char_to_int[char] for char in training_data]
    validation_data = [char_to_int[char] for char in validation_data]
    testing_data = [char_to_int[char] for char in testing_data]
    #

    print(len(training_data))
    print(len(validation_data))
    print(len(testing_data))
    train_d = torch.Tensor(list((training_data)))
    valid_d = torch.Tensor(list((validation_data)))
    test_d = torch.Tensor(list((testing_data)))
    return train_d, valid_d, test_d,data


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source,bptt,ntokens,model,criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data.long())
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data.long(), hidden)
                output = output.view(-1, ntokens)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets.long()).item()
    return total_loss / (len(data_source) - 1)

best_val_loss = None
lr = args.lr
def train(train_data,bptt,ntokens,model,criterion,optimizer,epoch):
    model.train()
    total_loss = 0.
    start_time = time.time()
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)

        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data.long())
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data.long(), hidden)
        loss = criterion(output.view(-1,ntokens), targets.long())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        #for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)
        loss = (loss + 0.3*(torch.pow(torch.sub(model.encoder.weight, model.fc1def.weight), 2)).sum())
        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} '.format(
                epoch, batch, len(train_data) // bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()

eval_batch_size = 20
datasets = ['data_kinder', 'data_elsass', 'data_nova-scotia', 'data_schweiz',
             'data_shanxi', 'data_bach', 'data_osterrh', 'data_jogoslav']
context_len = [5,10,20,30,50]

fr = open('erbp_results.txt', 'w')

def main_func(datasets, context_len, epochs):
    for dataset in datasets:
        for bptt in context_len:
            train_d, valid_d, test_d, data = data_generator(dataset)
            train_data = batchify(train_d, args.batch_size)
            val_data = batchify(valid_d, eval_batch_size)
            test_data = batchify(test_d, eval_batch_size)

            ntokens = len(set(data))
            best_val_loss = None
            lr = args.lr
            if args.model == 'Transformer':
                model = model1.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
            else:
                model = model1.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), args.lr)

            for epoch in range(1, epochs+1):
                epoch_start_time = time.time()
                train(train_data, bptt,ntokens,model,criterion,optimizer,epoch)
                val_loss = evaluate(val_data, bptt,ntokens,model,criterion)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   val_loss, math.exp(val_loss)))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    lr /= 0.4

            # Load the best saved model.
            with open(args.save, 'rb') as f:
                model = torch.load(f)
                if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
                    model.rnn.flatten_parameters()
            # Run on test data.
            test_loss = evaluate(test_data,bptt,ntokens,model,criterion)
            print('=' * 89)
            print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                test_loss, math.exp(test_loss)))
            print('=' * 89)
            fr.writelines("test loss for len %d and dataset %s is %f\n" % (bptt,dataset, test_loss))
            #fr.close()
    return

main_func(datasets,context_len,args.epochs)


