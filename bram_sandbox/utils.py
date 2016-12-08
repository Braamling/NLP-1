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

def get_multi_hot(ingredients, ingredient_list):
    multi_hot = np.zeros(len(ingredient_list))

    for ingredient in ingredients:
        multi_hot[ingredient.values()[0]] = 0

    return multi_hot


'''
[
    recipe {
        ingredients: []
        steps: [{sentence: line1, "...": "..." }, {sentence: line1}, {sentence: line1}]
    }

]
'''
def get_ingredient_list_size(dict_fn):
    return len(load_pickle_to_dict(dict_fn))    

def get_words_from_dataset(fn):
    with open(fn) as recipe_file:    
        recipes = json.load(recipe_file)

        for recipe in recipes:
            recipe = recipe['steps']
            for step in recipe:
                for word in step['sentence'].split():
                    yield word
            yield '<endofrecipe>'

def get_dataset(fn, dict_fn, vocab):
    ingredient_list = load_pickle_to_dict(dict_fn)

    with open(fn) as recipe_file:    
        recipes = json.load(recipe_file)

        for recipe in recipes:
            ingredient_multi_hot = get_multi_hot(recipe['ingredients'], ingredient_list)
            recipe = recipe['steps']

            # Create one hot vectors for each word in the recipe 
            recipe = np.array([vocab.encode(word) for word in yield_words(recipe)])

            yield Recipe(ingredient_multi_hot, recipe)

def yield_words(recipe):
    for step in recipe:
        for word in step['sentence'].split():
            yield word
        yield '<endofrecipe>'

def load_pickle_to_dict(fn):
    return pickle.load(open(fn, 'rb'))

def recipe_iterator(recipes, batch_size, num_steps):
    len(recipes) 

def ptb_iterator(raw_data, batch_size, num_steps):
    # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
    

    # for recipe in raw
    # Create an numpy array of the complete list of words
    raw_data = np.array(raw_data, dtype=np.int32)

    # Get data length
    data_len = len(raw_data)

    # Get the amount of batches needed
    num_batches = data_len // batch_size

    # Create a placeholder with a row for each batch
    data = np.zeros([batch_size, num_batches], dtype=np.int32)


    # Fill the placeholder with all batches. 
    for i in range(batch_size):
        data[i] = raw_data[num_batches * i:num_batches * (i + 1)]
    epoch_size = (num_batches - 1) // num_steps



    # Just crash because stupid
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    # Retrieve each set of batches
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


class Recipe():

    def __init__(self, multihot, words):
        self.multihot = multihot
        self.words = words


    def get_multihot(self):
        return self.multihot

    def create_sequences(self, number_of_steps):
        padding_size = number_of_steps - (len(self.words) % number_of_steps)

        words = np.concatenate((self.words, np.zeros([padding_size + 1])), axis=0)

        self.sequences_x = np.split(words[:-1], number_of_steps)
        self.sequences_y = np.split(words[1:], number_of_steps)
