from collections import defaultdict

import numpy as np
import json
import pickle

class Vocab(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = '<unk>'
        self.add_word(self.unknown, count=0)

    def add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_freq)

def calculate_perplexity(log_probs):
    # https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
    perp = 0
    for p in log_probs:
        perp += -p
    return np.exp(perp / len(log_probs))

# def get_dataset(fn):
#     for line in open(fn):
#         for word in line.split():
#             yield word
#         yield '<eos>'

def get_dataset(fn):
    with open(fn) as recipe_file:    
        recipes = json.load(recipe_file)

        for recipe in recipes:
            for step in recipe:
                for word in step['sentence'].split():
                    yield word
            yield '<endofrecipe>'

def load_pickle_to_dict(fn):
    return pickle.load(open(fn, 'rb'))

def ptb_iterator(raw_data, batch_size, num_steps):
    # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)

def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))
