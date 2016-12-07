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

def get_dataset(fn, dict_fn, vocab, number_of_steps, batch_size):
    ingredient_list = load_pickle_to_dict(dict_fn)
    recipes = []
    with open(fn) as recipe_file:    
        recipes_json = json.load(recipe_file)

        for recipe in recipes_json:
            ingredient_multi_hot = get_multi_hot(recipe['ingredients'], ingredient_list)

            # Create one hot vectors for each word in the recipe 
            recipe = [vocab.encode(word) for word in yield_words(recipe['steps'])]
            recipe = np.array(recipe)
            recipes.append(Recipe(number_of_steps, ingredient_multi_hot, recipe))

    return [batch for batch in recipe_iterator(recipes, batch_size, number_of_steps)]

def yield_words(recipe):
    for step in recipe:
        for word in step['sentence'].split():
            yield word
        yield '<endofrecipe>'

def load_pickle_to_dict(fn):
    return pickle.load(open(fn, 'rb'))

def recipe_iterator(recipes, batch_size, number_of_steps):
    padding_size = batch_size - (len(recipes) % batch_size)

    # pad the recipes with empty recipes
    for i in range(padding_size):
        recipes.append(Recipe(number_of_steps))
    
    # Create batches for each set of reripces
    for recipes in np.split(np.asarray(recipes), batch_size):
        yield RecipeBatch(recipes)


def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))


class Recipe():

    def __init__(self, number_of_steps, multihot=None, words=None):
        if multihot is None:
            self.sequences_x = []
            self.sequences_y = []
        else:
            self.multihot = multihot
            self.words = words
            self.number_of_steps = number_of_steps
            self.create_sequences(number_of_steps)

    def get_multihot(self):
        return self.multihot

    def create_sequences(self, number_of_steps):
        padding_size = number_of_steps - (len(self.words) % number_of_steps)

        words = np.concatenate((self.words, np.zeros([padding_size + 1])), axis=0)

        # Create two sequences with a 1 index difference
        self.sequences_x = np.split(words[:-1], number_of_steps)
        self.sequences_y = np.split(words[1:], number_of_steps)

    def get_sequences_amount(self):
        return len(self.sequences_y)

    def get_sequence_i(self, index):
        if index >= (self.sequences_x):
            return np.zeros(number_of_steps), np.zeros(number_of_steps)
        else:
            return self.sequences_x[index], self.sequences_y[index]

class RecipeBatch():

    def __init__(self, recipes):
        self.recipes = recipes
        self.max_seq = 0

        for recipe in recipes:
            if recipe.get_sequences_amount() > self.max_seq:
                self.max_seq = recipe.get_sequences_amount()

    def get_all_sequence_i(self, index):
        batch_x = []
        batch_y = []

        for recipe in recipes:
            x, y = recipe.get_sequence_i(index)
            batch_x.append(x)
            batch_y.append(y)
        
        return (batch_x, batch_y)

    def get_all_multihots(self):
        multihots = []

        for recipe in recipes:
            multihots.append(recipe.get_multihot())

        return multihots

    def get_max_sequence_size(self):
        return self.max_seq



