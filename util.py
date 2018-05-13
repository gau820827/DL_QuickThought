"""Some useful utilizations. Borrowed from Pytorch Tutorial."""
import time
import math

import torch

PAD_TOKEN = 2
UNK_TOKEN = 3
BLK_TOKEN = 4

use_cuda = torch.cuda.is_available()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def gettime(start):
    now = time.time()
    return asMinutes(now - start)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def load_model(model, model_src, mode='eval'):
    state_dict = torch.load(model_src, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    if mode == 'eval':
        model.eval()
    else:
        model.train()

    return model


def add_sentence_paddings(summarizes):
    """A helper function to add paddings to sentences.
    Args:
        summary: A list (batch_size) of indexing summarizes.
                 [tokens]
        padding index = 2
    Returns:
        A list (batch_size) with padding summarizes.
    """
    # Add block paddings
    def len_block(summary):
        return summary.count(BLK_TOKEN)

    max_blocks_length = max(list(map(len_block, summarizes)))

    for i in range(len(summarizes)):
        summarizes[i] += [BLK_TOKEN for j in range(max_blocks_length - len_block(summarizes[i]))]

    # Aligns with blocks, and remove <BLK> at this time
    def to_matrix(summary):
        mat = [[] for i in range(len_block(summary))]
        idt = 0
        for word in summary:
            if word == BLK_TOKEN:
                idt += 1
            else:
                mat[idt].append(word)
        return mat

    for i in range(len(summarizes)):
        summarizes[i] = to_matrix(summarizes[i])

    # Add sentence paddings
    def len_sentence(matrix):
        return max(list(map(len, matrix)))

    max_sentence_length = max([len_sentence(s) for s in summarizes])
    for i in range(len(summarizes)):
        for j in range(len(summarizes[i])):
            summarizes[i][j] += [PAD_TOKEN for k in range(max_sentence_length - len(summarizes[i][j]))]
            summarizes[i][j] += [BLK_TOKEN]

    # Join back the matrix
    def to_list(matrix):
        return [j for i in matrix for j in i]

    for i in range(len(summarizes)):
        summarizes[i] = to_list(summarizes[i])

    return summarizes, max_sentence_length + 1, max_blocks_length


def addpaddings(tokens):
    """A helper function to add paddings to tokens.

    Args:
        summary: A list (batch_size) of indexing tokens.

    Returns:
        A list (batch_size) with padding tokens.
    """
    max_length = len(max(tokens, key=len))
    for i in range(len(tokens)):
        tokens[i] += [PAD_TOKEN for i in range(max_length - len(tokens[i]))]
    return tokens
