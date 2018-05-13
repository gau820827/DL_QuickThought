"""This is the file for prepare yelp review dataset."""
import json
import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict

from dataprepare import Lang, YELP

PAD_TOKEN = 2
UNK_TOKEN = 3
BLK_TOKEN = 4

# Use NLTK tokenizer
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = WordPunctTokenizer()

word_freq = defaultdict(int)

# Build the vocabulary sets
with open('./data/yelp_academic_dataset_review.json', 'rb') as f:
    for line in f:
        review = json.loads(line)
        words = word_tokenizer.tokenize(review['text'])
        for word in words:
            word_freq[word] += 1

    print("load finished")

# Save the vocabulary frequency
with open('word_freq.pickle', 'wb') as g:
    pickle.dump(word_freq, g)
    print(len(word_freq))  # 159654
    print("word_freq save finished")

num_classes = 5
sort_words = list(sorted(word_freq.items(), key=lambda x: -x[1]))

# Replace the words that appear 5 times with a <UNK> token
yelp_lang = Lang('yelp')
for word, freq in word_freq.items():
    if freq <= 5:
        yelp_lang.addword('<UNK>')
    else:
        yelp_lang.addword(word)

# Save the Lang
with open('yelp_lang.pickle', 'wb') as g:
    pickle.dump(yelp_lang, g)
    print("yelp_lang save finished")

# Build the indexing version data
data = []

with open('./data/yelp_academic_dataset_review.json', 'rb') as f:
    n = 0
    for line in f:
        yelp = YELP()
        words = []
        idx_words = []

        # Load the review
        review = json.loads(line)
        sents = sent_tokenizer.tokenize(review['text'])
        for ids, sent in enumerate(sents):
            for idw, word in enumerate(word_tokenizer.tokenize(sent)):
                words.append(word)
                idx_words.append(yelp_lang.word2index.get(word, UNK_TOKEN))
            words.append('<BLK>')
            idx_words.append(BLK_TOKEN)

        # Store the data
        yelp.words = words
        yelp.idx_words = idx_words
        yelp.label = int(review['stars'])
        yelp.sent_leng = len(sents)

        data.append(yelp)

        # n += 1
        # if (n >= 10000):
        #     break

    # Dump to a pickle file
    pickle.dump(data, open('yelp_data.pickle', 'wb'))
    print('Parse {} reviews'.format(len(data)))  # 229907
