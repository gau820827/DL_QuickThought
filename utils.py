import torch
# import json

def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        test1 = torch.sum((mat ** 2), 1)
        test2 = torch.sum(test1, 1)
        # test2 = torch.sum(test1, 2)
        ret = (test2.squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')

'''
def package(data, volatile=False):
    """Package data for training / evaluation."""
    data = map(lambda x: json.loads(x), data)
    dat = map(lambda x: map(lambda y: emb.word2idx[y], x['text']), data)
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = map(lambda x: x['label'], data)
    maxlen = min(maxlen, 500)
    for i in range(len(data)):
        if maxlen < len(dat[i]):
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(emb.word2idx['<pad>'])
    dat = Variable(torch.LongTensor(dat), volatile=volatile)
    targets = Variable(torch.LongTensor(targets), volatile=volatile)
    return dat.t(), targets
'''
