from collections import defaultdict

import numpy as np
import json
import random
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

    def encode_list(self, list):
        encoded = []
        for word in list:
            encoded.append({word: self.encode(word)})

        return encoded

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_freq)

class Recipe():
    def __init__(self, number_of_steps, multihot=None, words=None):
        if words is None:
            self.sequences_x = []
            self.sequences_y = []
            self.number_of_steps = number_of_steps
            self.multihot = multihot
            self.words = None
        else:
            self.multihot = multihot
            self.words = words
            self.number_of_steps = number_of_steps
            self.create_sequences(number_of_steps)

    """
    Get the multihot ingredient list of this recipe
    """
    def get_multihot(self):
        return self.multihot

    """
    Create the sequences of length "number_of_steps" and pad the last one with
    zeros
    """
    def create_sequences(self, number_of_steps):
        padding_size = number_of_steps - (len(self.words) % number_of_steps)

        words = np.concatenate((self.words, np.zeros([padding_size + 1])), axis=0)
        # Create two sequences with a 1 index difference
        self.sequences_x = split(words[:-1], number_of_steps)
        self.sequences_y = split(words[1:], number_of_steps)

    """
    Get the actual amount of sequences of length "number_of_steps" of this recipe
    """
    def get_sequences_amount(self):
        return len(self.sequences_y)


    """
    Get the sequence x and y on position i. two zero arrays of length 
    "number_of_steps" is returned if the index is out of range.
    """
    def get_sequence_i(self, index):
        if index >= len(self.sequences_x):
            return np.zeros(self.number_of_steps), np.zeros(self.number_of_steps)
        else:
            return self.sequences_x[index], self.sequences_y[index]

class RecipeBatch():
    """
    A recipe batch consists of N Recipe object that can be used for iteration.
    """

    def __init__(self, recipes):
        self.recipes = recipes
        self.max_seq = 0

        # Find the histest amount of sequences in the batch.
        for recipe in recipes:
            if recipe.get_sequences_amount() > self.max_seq:
                self.max_seq = recipe.get_sequences_amount()

    """
    Retrieve a tupel with two all x and y sequences of this batch
    """
    def get_all_sequence_i(self, index):
        batch_x = []
        batch_y = []

        for recipe in self.recipes:
            x, y = recipe.get_sequence_i(index)
            batch_x.append(x)
            batch_y.append(y)
        
        return batch_x, batch_y

    def get_all_recipes(self):
        return self.recipes

    """
    Retrieve an array with all multihots of all sequences in this batch
    """
    def get_all_multihots(self):
        multihots = []

        for recipe in self.recipes:
            multihots.append(recipe.get_multihot())

        return multihots

    """
    Get the largest sequence size in this batch
    """
    def get_max_sequence_size(self):
        return self.max_seq


def calculate_perplexity(log_probs):
    # https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
    perp = 0
    for p in log_probs:
        perp += -p
    return np.exp(perp / len(log_probs))



def get_ingredient_list_size(dict_fn):
    return len(load_pickle_to_dict(dict_fn))    


def get_random_multihot(ingredient_list_size, vocab):
    multi_hot = np.zeros((1, ingredient_list_size))

    indices = random.sample(range(ingredient_list_size), 15)
    for i in indices:
        multi_hot[0][i] = 1

    return multi_hot

"""
Yield all words in a dataset
"""
def get_words_from_dataset(fn):
    with open(fn) as recipe_file:    
        recipes = json.load(recipe_file)

        for recipe in recipes:
            recipe = recipe['steps']
            for step in recipe:
                for word in step['sentence'].split():
                    yield word
            yield '<endofrecipe>'

"""
Retrieve that dataset as a list of Recipe batches.
"""
def get_dataset(fn, dict_fn, vocab, number_of_steps, batch_size):
    ingredient_list = load_pickle_to_dict(dict_fn)
    multihot_size = len(ingredient_list)
    recipes = []
    with open(fn) as recipe_file:    
        recipes_json = json.load(recipe_file)
        total_ingredients = {}

        # Create a recipe object for each recipe.
        for recipe in recipes_json:

            ingredient_multi_hot = get_multi_hot(recipe['ingredients'], ingredient_list)
            for ingredient in recipe['ingredients']:
                total_ingredients[ingredient.keys()[0]] = ingredient.values()[0]
            # Create one hot vectors for each word in the recipe 
            recipe = [vocab.encode(word) for word in yield_words(recipe['steps'])]
            recipe = np.array(recipe)
            recipes.append(Recipe(number_of_steps, ingredient_multi_hot, recipe))

    # print total_ingredients

    pickle.dump( total_ingredients, open( "list_of_foods.p", "wb" ) )

    # Create batches of recipes to later iterate
    recipe_batches = [batch for batch in
                      recipe_iterator(recipes, batch_size, number_of_steps, multihot_size)]

    return recipe_batches

"""
Return a multihot list based on the ingredient_list of a recipe
"""
def get_multi_hot(ingredients, ingredient_list):
    multi_hot = np.zeros((len(ingredient_list)))

    # Set all the multihot indecies to 1 of existing ingredients.
    for ingredient in ingredients:
        multi_hot[ingredient.values()[0]] = 1


    return multi_hot

def get_ingredients(multihot, ingredient_list):
    indices = np.where(multihot == 1)[0]
    ingredients = [ingredient_list[x] for x in indices]

    return ingredients
"""
Iterate over all given recipes and yield batches of recipes in a RecipeBatch object
"""
def recipe_iterator(recipes, batch_size, number_of_steps, multihot_size):
    padding_size = batch_size - (len(recipes) % batch_size)
    multihot_placeholder = np.zeros(multihot_size)
    # pad the recipes with empty recipes
    for i in range(padding_size):
        recipes.append(Recipe(number_of_steps, multihot_placeholder))
    
    # Create batches for each set of reripces
    for recipes in split(np.asarray(recipes), batch_size):
        yield RecipeBatch(recipes)

"""
Yield all the words in a recipe into a single sequence.
"""
def yield_words(recipe):
    for step in recipe:
        for word in step['sentence'].split():
            yield word
        yield '<endofrecipe>'

"""
Load a pickle file in dict form into memory
"""
def load_pickle_to_dict(fn):
    return pickle.load(open(fn, 'rb'))


def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))


def split(data, n_steps):
    return [data[y:y+n_steps] for y in xrange(0, len(data), n_steps)]