"""This is the file for main model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from util import use_cuda


class AttnRNN(nn.Module):
    """The module for non-hierarchical model."""

    def __init__(self, embedding_size, hidden_size, vocab_size, output_size):
        super(AttnRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.GRU(embedding_size, hidden_size // 2, bidirectional=True)
        self.out = nn.Linear(hidden_size * 2, output_size)

        self.init_weights()

    def forward(self, inputs, sent_length, block_length):
        # Embedding
        embedded = self.embedding(inputs)

        # Encoding
        embedded = embedded.permute(1, 0, 2)
        outputs, hidden = self.encoder(embedded)
        # outputs (seq_len, batch, hidden_size * num_directions)
        # hidden  (num_layers * num_directions, batch, hidden_size)

        # Final output
        hidden = torch.cat((outputs[0, :, :], outputs[-1, :, :]), dim=1)
        output = F.softmax(self.out(hidden), dim=1)
        return output

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.out]
        em_layer = [self.embedding]

        for layer in lin_layers + em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)


class HierarchicalAttnRNN(nn.Module):
    """The module for heirarchical attention."""

    def __init__(self, embedding_size, hidden_size, vocab_size, output_size):
        super(HierarchicalAttnRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.LocalEncoder = EncoderRNN(embedding_size, hidden_size, level='local')
        self.LocalAttn = LocalAttn(hidden_size)
        self.GlobalEncoder = EncoderRNN(hidden_size, hidden_size, level='global')
        self.GlobalAttn = GlobalAttn(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.init_weights()

        # Configurations
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

    def forward(self, inputs, sent_length, block_length):
        # Configurations
        batch_length = inputs.size(0)
        input_length = inputs.size(1)
        hidden_size = self.hidden_size

        # Reshape the input to block-batch
        inputs = inputs.view(batch_length * block_length, sent_length)

        # Embedding
        embedded = self.embedding(inputs)

        # Word-level Encoding
        local_encoder_outputs, local_hidden = self.LocalEncoder(embedded, sent_length,
                                                                block_length)

        # Reshape back to (batch_size * blk, sent_length, hidden_size)
        local_encoder_outputs = local_encoder_outputs.permute(1, 0, 2)
        global_input, l_attn_weights = self.LocalAttn(local_encoder_outputs,
                                                      sent_length, block_length)

        # Sent-level Encoding
        global_encoder_outputs, global_hidden = self.GlobalEncoder(global_input,
                                                                   sent_length, block_length)

        # Sent-level Attention
        global_encoder_outputs = global_encoder_outputs.permute(1, 0, 2)
        context_vector, g_attn_weights = self.GlobalAttn(global_encoder_outputs)

        # Final Classifier
        output = F.softmax(self.out(context_vector), dim=1)
        return output

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.out]
        em_layer = [self.embedding]

        for layer in lin_layers + em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)


class EncoderRNN(nn.Module):
    """Vanilla encoder using pure gru."""
    def __init__(self, embedding_size, hidden_size, level='local'):
        super(EncoderRNN, self).__init__()
        self.level = level
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size // 2, bidirectional=True)

    def forward(self, embedded, sent_length, block_length):
        # embedded is of size (n_batch, seq_len, emb_dim)
        # gru needs (seq_len, n_batch, emb_dim)
        if self.level == 'local':
            inp = embedded.permute(1, 0, 2)  # (seq_len, batch, emb_dim)

            # To GRU module
            outputs, hidden = self.gru(inp)

            # size of outputs: (sent_length, batch * blk, hidden_size)

        else:
            embedded = embedded.permute(1, 0, 2)
            outputs, hidden = self.gru(embedded)
        # outputs (seq_len, batch, hidden_size * num_directions)
        # hidden  (num_layers * num_directions, batch, hidden_size)
        return outputs, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size), requires_grad=False)

        if use_cuda:
            return result.cuda()
        else:
            return result


class LocalAttn(nn.Module):
    """The module for word-level attention."""

    def __init__(self, hidden_size):
        super(LocalAttn, self).__init__()
        self.attn = Attn(hidden_size)
        self.uw = nn.Parameter(torch.FloatTensor(1, hidden_size).uniform_(-1, 1))

    def forward(self, encoder_outputs, sent_length, block_length):
        # Configuration
        batch_size = encoder_outputs.size(0) // block_length
        hidden_size = encoder_outputs.size(2)

        # calculate attention scores for each block
        hiddens = self.uw.repeat(batch_size * block_length, 1)

        attn_weights = self.attn(hiddens, encoder_outputs)

        block_context = torch.bmm(attn_weights, encoder_outputs)  # (batch * blk, 1, hid)
        block_context = block_context.view(batch_size, block_length, hidden_size)

        return block_context, attn_weights


class GlobalAttn(nn.Module):
    """The module for sentence-level attention."""

    def __init__(self, hidden_size):
        super(GlobalAttn, self).__init__()
        self.attn = Attn(hidden_size)
        self.us = nn.Parameter(torch.FloatTensor(1, hidden_size).uniform_(-1, 1))

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        hidden = self.us.repeat(batch_size, 1)

        attn_weights = self.attn(hidden, encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs)

        # Adjust the dimension after bmm()
        context = context.squeeze(1)

        return context, attn_weights


class Attn(nn.Module):
    """ The score function for the attention mechanism.

    We define the score function as the dot product function from Luong et al.
    Where score(s_{i}, h_{j}) = s_{i} T h_{j}

    """
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden, encoder_outputs):
        batch_size, seq_len, hidden_size = encoder_outputs.size()

        # Get hidden chuncks (batch_size, seq_len, hidden_size)
        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        hiddens = hidden.repeat(1, seq_len, 1)
        attn_energies = self.score(hiddens, encoder_outputs)

        # Normalize energies to weights in range 0 to 1, resize to B x 1 x seq_len
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # print('size of hidden: {}'.format(hidden.size()))
        # print('size of encoder_hidden: {}'.format(encoder_output.size()))
        energy = self.attn(encoder_outputs)
        energy = F.tanh(energy)

        # batch-wise calculate dot-product
        hidden = hidden.unsqueeze(2)  # (batch, seq, 1, d)
        energy = energy.unsqueeze(3)  # (batch, seq, d, 1)

        energy = torch.matmul(hidden, energy)  # (batch, seq, 1, 1)

        # print('size of energies: {}'.format(energy.size()))

        return energy.squeeze(3).squeeze(2)
