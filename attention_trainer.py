# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import importlib
import data
import utils
import time
import json
import models



# This is the iterator we'll use during training.
# It's a generator that gives you one batch at a time.
def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)
        batch_indices = order[start:start + batch_size]
        yield [source[index] for index in batch_indices]

# This is the iterator we use when we're evaluating our model.
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        if (len(batch) != batch_size):
            break
        batches.append(batch)

    return batches

# The following function gives batches of vectors and labels,
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    labels = []
    for dict in batch:
        vectors.append(dict["text_index_sequence"])
        labels.append(dict["label"])
    return vectors, labels


def evaluate(model, data_iter):
    model.eval()
    correct = 0
    total = 0
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = Variable(torch.stack(vectors).squeeze())
        labels = torch.stack(labels).squeeze()
        output = model(vectors)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        if (predicted.shape != labels.shape):
            break
        correct += (predicted.cpu() == labels.cpu()).sum()
        # correct += (predicted == labels).sum()
    return correct / float(total)

def training_loop(model, criterion, optimizer, training_iter, dev_iter, train_eval_iter, use_gpu=False):
    step = 0
    for i in range(num_train_steps):
        model.train()
        vectors, labels = get_batch(next(training_iter))
        vectors = Variable(torch.stack(vectors).squeeze())
        labels = Variable(torch.stack(labels).squeeze())
        if (use_gpu):
            vectors = vectors.cuda()
            labels = labels.cuda()

        model.zero_grad()
        output = model(vectors)

        loss = criterion(output, labels)
        loss.backward(retain_graph=True)
        # loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print( "Step %i; Loss %f; Train acc: %f; Dev acc %f"
                %(step, loss.data[0], evaluate(model, train_eval_iter), evaluate(model, dev_iter)))

        step += 1

def train(model, use_gpu):
    model = model

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    training_iter = data_iter(dataset.training_set, batch_size)
    train_eval_iter = eval_iter(dataset.training_set[0:500], batch_size)
    dev_iter = eval_iter(dataset.dev_set[0:500], batch_size)
    training_loop(model, criterion, optimizer, training_iter, dev_iter, train_eval_iter, use_gpu)
    return model

def test(model):
    # Test the model
    test_iter = eval_iter(dataset.test_set, batch_size)
    test_acc = evaluate(model, test_iter)
    print('Accuracy of the network on the test data: %f' % (100 * test_acc))


# Set the random seed manually for reproducibility.
# seed = 1111
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     if not cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#     else:
#         torch.cuda.manual_seed(seed)
# random.seed(seed)


def training_loop2(model, criterion, optimizer, training_iter, dev_iter, train_eval_iter, step, total_pure_loss, total_loss, cuda=False):
    for i in range(num_train_steps):
        model.train()
        data, targets = get_batch(next(training_iter))
        data = Variable(torch.stack(data).squeeze())
        targets = Variable(torch.stack(targets).squeeze())
        if (cuda):
            data = data.cuda()
            targets = targets.cuda()
        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        # print ("attention:", attention)
        loss = criterion(output.view(data.size(1), -1), targets)
        total_pure_loss += loss.data

        if attention is not None:  # add penalization term
            attentionT = torch.transpose(attention, 1, 2).contiguous()
            bmm_result = torch.bmm(attention, attentionT)
            I_for_loss = I[:attention.size(0)]
            extra_loss = utils.Frobenius(bmm_result - I_for_loss)
            loss += penalization_coeff * extra_loss
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.data

        if step % 100 == 0:
            # elapsed = time.time() - start_time
            print(
                   "Step",  step,
                   "Loss ", loss.data[0],
                   " Train loss/acc: ", str(evaluate2(model, train_eval_iter)),
                   "Dev loss/acc: " + str(evaluate2(model, dev_iter))
                 )
            # print( "Step",  step)
            # print ("Loss ", loss.data[0])
            # print (" Train loss/acc: ", str(evaluate2(model, train_eval_iter)))
            # print ("Dev loss/acc: " + str(evaluate2(model, dev_iter)))
            # print('| {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | pure loss {:5.4f}'.format(
            #        i, len(data_train) // batch_size,
            #       elapsed * 1000 / log_interval, total_loss[0] / log_interval,
            #       total_pure_loss[0] / log_interval))
            total_loss = 0
            total_pure_loss = 0
            # start_time = time.time()
        step += 1

        """
        if i % log_interval == 0 and i > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | pure loss {:5.4f}'.format(
                  epoch_number, i, len(data_train) // batch_size,
                  elapsed * 1000 / log_interval, total_loss[0] / log_interval,
                  total_pure_loss[0] / log_interval))
            total_loss = 0
            total_pure_loss = 0
            start_time = time.time()
        """

#            for item in model.parameters():
#                print item.size(), torch.sum(item.data ** 2), torch.sum(item.grad ** 2).data[0]
#            print model.encoder.ws2.weight.grad.data
#            exit()

def test2(model, epoch_number, evaluate_start_time):
    # Test the model
    test_iter = eval_iter(dataset.test_set, batch_size)
    val_loss, test_acc = evaluate2(model, test_iter)
    print('-' * 89)
    fmt = '| evaluation | epoch: {} |time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
    print(fmt.format(epoch_number, (time.time() - evaluate_start_time), val_loss, test_acc))
    print('-' * 89)
    # print('Accuracy of the network on the test data: %f' % (100 * test_acc))
    return val_loss, test_acc

def save_model(model, val_loss, acc, epoch_number):
    # Save the model, if the validation loss is the best we've seen so far.
    global best_val_loss, best_acc
    save = "models/model-medium.pt"
    if not best_val_loss or val_loss < best_val_loss:
        with open(save, 'wb') as f:
            torch.save(model, f)
        f.close()
        best_val_loss = val_loss
    # else:  # if loss doesn't go down, divide the learning rate by 5.
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = param_group['lr'] * 0.2
    if not best_acc or acc > best_acc:
        with open(save[:-3]+'.best_acc.pt', 'wb') as f:
            torch.save(model, f)
        f.close()
        best_acc = acc
    with open(save[:-3]+'.epoch-{:02d}.pt'.format(epoch_number), 'wb') as f:
        torch.save(model, f)
    f.close()


def train2(model, epoch_number, cuda=True):
    model.train()
    total_loss = 0
    total_pure_loss = 0  # without the penalization term
    start_time = time.time()
    step = 0


    training_iter = data_iter(dataset.training_set, batch_size)
    train_eval_iter = eval_iter(dataset.training_set[0:500], batch_size)
    dev_iter = eval_iter(dataset.dev_set[0:500], batch_size)

    training_loop2(model, criterion, optimizer, training_iter, dev_iter, train_eval_iter, step, total_pure_loss, total_loss, cuda)

    evaluate_start_time = time.time()
    val_loss, acc = test2(model, epoch_number, evaluate_start_time)
    save_model(model, val_loss, acc, epoch_number)


def evaluate2(model, data_iter):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    total_count = 0
    criterion = nn.CrossEntropyLoss()
    # for batch, i in enumerate(range(0, len(data_val), batch_size)):
    #     data, targets = package(data_val[i:min(len(data_val), i+batch_size)], volatile=True)
    for i in range(len(data_iter)):
        data, targets = get_batch(data_iter[i])
        data = Variable(torch.stack(data).squeeze())
        targets = Variable(torch.stack(targets).squeeze())
        if cuda:
            data = data.cuda()
            targets = targets.cuda()
        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        curr_loss = criterion(output_flat, targets)
        total_loss += curr_loss.data
        total_count += targets.size(0)
        prediction = torch.max(output_flat, 1)[1]
        total_correct += torch.sum((prediction == targets).float())
    return total_loss[0] / (total_count // batch_size), total_correct.data[0] / total_count # (ajusted_loss, acc)

if __name__ == '__main__':

    emb = data.Embedding()
    dataset = data.Dataset(emb.word2idx)

    # Hyper Parameters
    learning_rate = 0.004
    num_train_steps = 1000
    clip = 0.5
    log_interval = 100
    cuda = True
    batch_size = 20
    # attention_hops = 1
    penalization_coeff = 1.0


    default_config = {
            'dropout': 0.5,
            'input_size': emb.__len__(),
            'nlayers': 2,
            'hidden_dim': 300,
            'embedding_dim': 300,
            'pooling': 'mean',
            'attention-unit': 350,
            'attention-hops': 1,
            'nfc': 512,
            'dictionary': emb,
            'word-vector': "",
            'num_labels': 5,
            'cuda' : True,
            'batch_size' : 20
        }
    best_val_loss = None
    best_acc = None

    config = default_config
    # config["pooling"] = "all"
    model = models.Classifier(config)
    # WTF is this
    I = Variable(torch.ones(model.config['batch_size'], model.config['attention-hops'], model.config['attention-hops']))
    # for i in range(batch_size):
    #     for j in range(attention_hops):
    #         I.data[i][j][j] = 1
    if cuda:
        model = model.cuda()
        I = I.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # importlib.reload(utils)
    for epoch in range(50):
        train2(model, epoch)

