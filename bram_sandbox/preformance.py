from copy import deepcopy

import tensorflow as tf

from utils import get_dataset, Vocab, get_words_from_dataset, get_ingredients
from utils import load_pickle_to_dict
from rnn_model import RNNLM_Model
from generator import Text_Generator

from collections import Counter

import json
"""
Please set the 

"""


class Config(object):
    # Train and/or generate the model when running the script
    train = False
    generate = True 

    # Learning parameters
    batch_size = 200
    embed_size = 5
    hidden_size = 50
    ingredient_hidden_size = 30
    num_steps = 40
    max_epochs = 10
    early_stopping = 2
    dropout = 0.9
    lr = 0.005

    # Training data locations
    encoded_train = 'data/train.json'
    encoded_valid = 'data/valid.json'
    encoded_test = 'data/test.json'

    # All recipe data.
    merged_data = 'data/recipes.json'

    # Ingredients data
    ingredients_data = 'data/list_of_foods.p'

    # Session location and name for storing and loading stored model
    session_name = 'ptb_rnnlm_1.weights'
    store_location = './ptb_rnnlm_1.weights'

    def __init__(self):
        with open('config.json') as data_file:    
            config = json.load(data_file)

        self.session_name = config['session_name']
        self.store_location = config['store_location']

def generate_from(generator, starting_text, multihot, gen_model, ingredient_list):
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    # Open session and generate text
    with tf.Session() as session:  


        # Retrieved configured session.         
        saver.restore(session, generator.store_location)
        pizza = ""
        no_pizza = ""

        # ingredients = ["pizza"]
        ingredients = get_ingredients(multihot, ingredient_list)

        # ingredients = gen_model.vocab.encode_list(ingredients)
        # for i in range(1000):
        generated_text = (' '.join(generator.generate_sentence(
                session, starting_text=starting_text, 
                temp=1.0, multi_hot=multihot)))

        total_count = 0
        ingredients_count = 0.0
        for ingredient in ingredients:
            counted = generated_text.count(ingredient)
            total_count += counted
            if counted > 0:
                ingredients_count += 1.0

        count_percentage = (ingredients_count/len(ingredients)) * 100

        print "----START----"
        print str(count_percentage) + "% of the ingredients occured " + str(total_count) + " times in the text -----"
        print "Ingredients: " + str(ingredients) 
        print "Text: " + generated_text
        print "----END----"

        return {"text": generated_text, "ingredients": ingredients, "count": total_count, "percentage": count_percentage}



def main():
    # Create config for the training and generator models
    config = Config()
    gen_config = deepcopy(config)

    # Set the batch size and step size of the generator model to 1
    gen_config.batch_size = gen_config.num_steps = 1

    # Retrieve the ingredient_list and invert the dict
    ingredient_list = load_pickle_to_dict(config.ingredients_data)    
    ingredient_list = {v: k for k, v in ingredient_list.iteritems()}

    # We create the training model and generative model
    with tf.variable_scope('RNNLM') as scope:
        # if config.train:
        rnn_model = RNNLM_Model(config)

        # if config.generate:
        # This instructs gen_model to reuse the same variables as the model above
        scope.reuse_variables()
        gen_model = RNNLM_Model(gen_config)

    # Generate text based on the passed word.
    generator = Text_Generator(gen_config, gen_model)


    vocab = Vocab()
    vocab.construct(get_words_from_dataset(config.merged_data))
    encoded_test = [recipe for recipe in\
                         get_dataset(config.encoded_test,
                                     config.ingredients_data,
                                     vocab,
                                     config.num_steps,
                                     config.batch_size)]

    # start_text = "preheat"
    # ingredients = ["pizza"]
    results = []

    for batch in encoded_test: 
        for recipe in batch.get_all_recipes():
            start_text = vocab.decode(recipe.get_sequence_i(0)[0][0])
            results = generate_from(generator, start_text, recipe.get_multihot(), gen_model, ingredient_list)

if __name__=="__main__":
    main()