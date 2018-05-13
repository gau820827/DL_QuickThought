"""This is core training part, containing different models."""
import time
import random
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from model import HierarchicalAttnRNN
from util import gettime, load_model, use_cuda
from util import add_sentence_paddings, addpaddings

from dataprepare import Lang, YELP


PAD_TOKEN = 2
UNK_TOKEN = 3
BLK_TOKEN = 4


def data_iter(source, batch_size=32, shuffle=True):
    """The iterator to give batch data while training.

    Args:
        source: the source file to batchify
        batch_size: the batch_size

    Return:
        A generator to yeild batch from random order.
        Will start another random epoch while one epoch
        finished.
    """
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    if shuffle:
        random.shuffle(source)

    # Get batch based on similar length of sentences
    source.sort(key=lambda yelp: yelp.sent_leng)
    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            return

        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        yield batch


def get_batch(batch):
    """Get the batch into training format.

    Arg:
        batch: The iterator of the dataset

    Returns:
        batch_data: The origin processed data
                    (i.e batch_size * (words))
        batch_idx_data: The indexing-processed data
                        (e.g batch_size * (idx_words))
        batch_label: The labels
                     (e.g batch_size * (label))

    """
    batch_data = []
    batch_idx_data = []
    batch_label = []
    for d in batch:
        words, idx_words, label = d.get_data()
        batch_data.append(words)
        batch_idx_data.append(idx_words)
        batch_label.append(label)

    return batch_data, batch_idx_data, batch_label


def evaluate(model, data_iter):
    """The function for evaluating during training."""
    model.eval()
    correct = 0
    total = 0
    for dt in data_iter:
        words, idx_words, labels = get_batch(dt)

        idx_words, sent_leng, blk_leng = add_sentence_paddings(idx_words)

        # Transform to tensor
        idx_words = Variable(torch.LongTensor(idx_words), requires_grad=False)
        labels = Variable(torch.LongTensor(labels), requires_grad=False)
        labels = torch.add(labels, -1)

        if use_cuda:
            idx_words = idx_words.cuda()

        output = model(idx_words, sent_leng, blk_leng)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return correct.item() / float(total)


def train(train_set, valid_set, lang,
          batch_size=64, embedding_size=200,
          hidden_size=100, learning_rate=0.01, epoch_time=10,
          grad_clip=5, get_loss=10, save_model=5,
          output_file='HieAttn', pretrain=None, iter_num=None):
    """The training procedure."""
    # Set the timer
    start = time.time()

    # Initialize the model
    n_class = 5
    model = HierarchicalAttnRNN(embedding_size, hidden_size, lang.n_words,
                                n_class, batch_size)

    if use_cuda:
        model.cuda()

    # Choose optimizer
    # loss_optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    loss_optimizer = optim.Adagrad(model.parameters(), lr=learning_rate,
                                   lr_decay=0, weight_decay=0)

    # loss_optimizer = optim.Adam(model.parameters(),
    #                             lr=learning_rate)

    # Load pre-train model
    use_model = None
    if pretrain is not None and iter_num is not None:
        use_model = ['./models/' + pretrain + '_' + s + '_' + str(iter_num)
                     for s in ['model', 'optim']]

    if use_model is not None:
        model = load_model(model, use_model[0])
        loss_optimizer.load_state_dict(torch.load(use_model[1]))
        print("Load Pretrain Model {}".format(use_model))
    else:
        print("Not use Pretrain Model")

    criterion = nn.CrossEntropyLoss()

    # The training loop
    total_loss = 0
    iteration = 0
    for epo in range(1, epoch_time + 1):
        # Start of an epoch
        print("Epoch #%d" % (epo))

        # Get data
        train_iter = data_iter(train_set, batch_size=batch_size)
        for dt in train_iter:
            iteration += 1
            words, idx_words, labels = get_batch(dt)

            # For summary paddings, if the model is herarchical then pad between sentences
            # If the batch_size is 1 then we don't need to do sentence padding
            if batch_size != 1:
                idx_words, sent_leng, blk_leng = add_sentence_paddings(idx_words)
            else:
                idx_words = addpaddings(idx_words)

            # Transform to tensor
            idx_words = Variable(torch.LongTensor(idx_words), requires_grad=False)
            labels = Variable(torch.LongTensor(labels), requires_grad=False)
            labels = torch.add(labels, -1)

            if use_cuda:
                idx_words = idx_words.cuda()

            # Zero the gradient
            loss_optimizer.zero_grad()
            model.train()

            output = model(idx_words, sent_leng, blk_leng)
            loss = criterion(output, labels)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            loss_optimizer.step()

            total_loss += loss

            # Print the information and save model
            if iteration % get_loss == 0:
                print("Time {}, iter {}, loss = {:.4f}, train_acc = {}, dev_acc = {}".format(
                    gettime(start), iteration, total_loss / get_loss,
                    evaluate(model, data_iter(train_set[:50], batch_size=1)),
                    evaluate(model, data_iter(valid_set[:50], batch_size=1))))
                total_loss = 0

        if epo % save_model == 0:
            torch.save(model.state_dict(),
                       "models/{}_model_{}".format(output_file, iteration))
            torch.save(loss_optimizer.state_dict(),
                       "models/{}_optim_{}".format(output_file, iteration))
            print("Save the model at iter {}".format(iteration))

    return model


def read_dataset():
    """Read the dataset."""
    with open('sml_yelp_data.pickle', 'rb') as f:
        data = pickle.load(f)
        length = len(data)
        train_data = data[:int(length * 0.8)]
        valid_data = data[int(length * 0.8) + 1: int(length * 0.9)]
        test_data = data[int(length * 0.9) + 1:]

        print('Read {} training set'.format(len(train_data)))
        print('Read {} validation set'.format(len(valid_data)))
        print('Read {} test set'.format(len(test_data)))

    with open('yelp_lang.pickle', 'rb') as f:
        lang = pickle.load(f)
        print('Read yelp lang')

    return train_data, valid_data, test_data, lang


def main():
    """The main to start the training."""
    train_data, valid_data, test_data, yelp_lang = read_dataset()
    train(train_data, valid_data, yelp_lang)


if __name__ == '__main__':
    main()
