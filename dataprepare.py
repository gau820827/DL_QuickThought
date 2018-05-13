"""Classes for data preparation."""


class Lang:
    """A class to summarize encoding information.

    This class will build three dicts:
    word2index, word2count, and index2word for
    embedding information. Once a set of data is
    encoded, we can transform it to corrsponding
    indexing use the word2index, and map it back
    using index2word.

    Attributes:
        word2index: A dict mapping word to index.
        word2count: A dict mapping word to counts in the corpus.
        index2word: A dict mapping index to word.

    """

    def __init__(self, name):
        """Init Lang with a name."""
        # Ken added <EOB> on 04/04/2018
        self.name = name
        self.word2index = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3, "<BLK>": 4}
        self.word2count = {"<SOS>": 0, "<EOS>": 0, "<PAD>": 0, "<UNK>": 0, "<BLK>": 0}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>", 3: "<UNK>", 4: "<BLK>"}
        self.n_words = len(self.word2index)

    def addword(self, word):
        """Add a word to the dict."""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class YELP:
    """ This is the class for storing the IMDB dataset.

    There are four attributes in this class.

    Attributes:
        words:   A list storing the tokens of the comment
        idx_words:  The indexing version of the comment
        label: The label for the comment
        sent_leng: The length of sentences of the comment

    """
    def __init__(self):
        """Init an IMDB data storage."""
        self.words = []
        self.idx_words = []
        self.label = []
        self.sent_leng = 0

    def get_data(self):
        return self.words, self.idx_words, self.label
