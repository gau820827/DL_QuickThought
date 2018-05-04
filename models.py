from __future__ import print_function
import torch
from torch.autograd import Variable
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class BiLSTM(nn.Module):

    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config
        self.drop = nn.Dropout(config['dropout'])
        self.encoder = nn.Embedding(config['input_size'], config['embedding_dim'])
        self.bilstm = nn.LSTM(config['embedding_dim'], config['hidden_dim'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=True)
        self.nlayers = config['nlayers']
        self.hidden_dim = config['hidden_dim']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']
        # self.init_weights()
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
        if os.path.exists(config['word-vector']):
            print('Loading word vectors from', config['word-vector'])
            vectors = torch.load(config['word-vector'])
            assert vectors[2] >= config['embedding_dim']
            vocab = vectors[0]
            vectors = vectors[1]
            loaded_cnt = 0
            for word in self.dictionary.word2idx:
                if word not in vocab:
                    continue
                real_id = self.dictionary.word2idx[word]
                loaded_id = vocab[word]
                self.encoder.weight.data[real_id] = vectors[loaded_id][:config['embedding_dim']]
                loaded_cnt += 1
            print('%d words from external word vectors loaded.' % loaded_cnt)

    # note: init_range constraints the value of initial weights
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        emb = self.drop(self.encoder(inp))
        outp = self.bilstm(emb, hidden)[0]
        if self.pooling == 'mean':
            outp = torch.mean(outp, 0).squeeze()
        elif self.pooling == 'max':
            outp = torch.max(outp, 0)[0].squeeze()
        elif self.pooling == 'all' or self.pooling == 'all-word':
            outp = torch.transpose(outp, 0, 1).contiguous()
        # self.hidden = repackage_hidden(emb)
        self.hidden = outp
        return outp, emb

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * 2, bsz, self.hidden_dim).zero_()),
                Variable(weight.new(self.nlayers * 2, bsz, self.hidden_dim).zero_()))


class SelfAttentiveEncoder(nn.Module):

    def __init__(self, config):
        super(SelfAttentiveEncoder, self).__init__()
        self.config = config
        self.bilstm = BiLSTM(config)
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['hidden_dim'] * 2, config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dictionary = config['dictionary']
        # self.init_weights()
        self.attention_hops = config['attention-hops']

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        outp = self.bilstm.forward(inp, hidden)[0]
        size = outp.size()  # [bsz, len, hidden_dim]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, hidden_dim*2]
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == self.dictionary.word2idx['<pad>']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)





class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        if config['pooling'] == 'mean' or config['pooling'] == 'max':
            self.encoder = BiLSTM(config)
            self.fc = nn.Linear(config['hidden_dim'] * 2, config['nfc'])
        elif config['pooling'] == 'all':
            self.encoder = SelfAttentiveEncoder(config)
            self.fc = nn.Linear(config['hidden_dim'] * 2 * config['attention-hops'], config['nfc'])
        else:
            raise Exception('Error when initializing Classifier')
        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.pred = nn.Linear(config['nfc'], config['num_labels'])
        self.dictionary = config['dictionary']
        # self.init_weights()

    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)

    def forward(self, inp, hidden):
        outp, attention = self.encoder.forward(inp, hidden)
        outp = outp.view(outp.size(0), -1)
        fc = self.tanh(self.fc(self.drop(outp)))
        pred = self.pred(self.drop(fc))
        if type(self.encoder) == BiLSTM:
            attention = None
        return pred, attention

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

    def encode(self, inp, hidden):
        return self.encoder.forward(inp, hidden)[0]


# A Multi-Layer Perceptron (MLP)
class MLPClassifier(nn.Module):

    def __init__(self, config):
        super(MLPClassifier, self).__init__()
        self.config = config

        self.embed = nn.Embedding(config['input_size'], config['embedding_dim'], padding_idx=0)
        self.dropout = nn.Dropout(config['dropout'])

        self.linear_1 = nn.Linear(config['embedding_dim'], config['hidden_dim'])
        self.linear_2 = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.linear_3 = nn.Linear(config['hidden_dim'], config['num_labels'])
        self.init_weights()

    def forward(self, x):
        out = self.embed(x)
        out = self.dropout(out)
        out = torch.sum(out, dim=1)
        out = F.relu(self.linear_1(out))
        out = F.relu(self.linear_2(out))
        out = self.dropout(self.linear_3(out))
        return F.log_softmax(out)

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear_1, self.linear_2]
        em_layer = [self.embed]

        for layer in lin_layers+em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)


class LSTMSentiment(nn.Module):

    def __init__(self, config):
        super(LSTMSentiment, self).__init__()
        self.config = config
        self.hidden_dim = config['hidden_dim']
        self.use_gpu = config['cuda']
        self.batch_size = config['batch_size']
        self.dropout = config['dropout']
        self.embeddings = nn.Embedding(config['input_size'], config['embedding_dim'], padding_idx=0)
        self.lstm = nn.LSTM(input_size=config['embedding_dim'], hidden_size=config['hidden_dim'])
        self.hidden2label = nn.Linear(config['hidden_dim'], config['num_labels'])
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        if self.use_gpu:
            x = self.embeddings(sentence.cuda()).view(len(sentence), self.batch_size, -1)
        else:
            x = self.embeddings(sentence).view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        self.hidden = repackage_hidden(self.hidden)
        log_probs = F.log_softmax(y)
        return log_probs

class BiLSTMSentiment(nn.Module):

    def __init__(self, config):
        super(BiLSTMSentiment, self).__init__()
        self.config = config
        self.hidden_dim = config['hidden_dim']
        self.use_gpu = config['cuda']
        self.batch_size = config['batch_size']
        self.dropout = config['dropout']
        self.embeddings = nn.Embedding(config['input_size'], config['embedding_dim'], padding_idx=0)
        self.lstm = nn.LSTM(input_size=config['embedding_dim'], hidden_size=config['hidden_dim'], bidirectional=True)
        self.hidden2label = nn.Linear(config['hidden_dim']*2, config['num_labels'])
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        x = self.embeddings(sentence.cuda()).view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        self.hidden = repackage_hidden(self.hidden)
        return log_probs
