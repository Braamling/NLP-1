from copy import deepcopy

import tensorflow as tf

from utils import get_dataset, Vocab, get_words_from_dataset, get_ingredients
from utils import load_pickle_to_dict
from rnn_model_word2vec import Word2VecLSTM
from rnn_model_embedding import EmbeddingLSTM
from generator import Text_Generator

from collections import Counter

import json
import csv
"""
Please set the 

"""


class Config(object):
    # Train and/or generate the model when running the script
    train = True
    generate = True
    use_word2vec = True

    # Learning parameters
    batch_size = 40
    embed_size = 50
    hidden_size = 50
    ingredient_hidden_size = 30
    num_steps = 100
    max_epochs = 1000
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
    session_name = 'withPCA.weights'
    store_location = './withPCA.weights'
    # session_name = 'ptb_rnnlm_1.weights'
    # store_location = './ptb_rnnlm_1.weights'

    def __init__(self):
        with open('config.json') as data_file:    
            config = json.load(data_file)

        self.session_name = config['session_name']
        self.store_location = config['store_location']

def generate_from(generator, starting_text, multihot, pca, gen_model, ingredient_list, session):
        # Retrieved configured session.         
        
        pizza = ""
        no_pizza = ""

        # ingredients = ["pizza"]
        # print ingredient_list
        print "2"
        ingredients = get_ingredients(multihot, ingredient_list)
        # print ingredients

        # ingredients = gen_model.vocab.encode_list(ingredients)
        # for i in range(1000):
        print "3"
        generated_text = (' '.join(generator.generate_sentence(
                session, starting_text=starting_text, 
                temp=1.0, multi_hot=multihot, pca_ingredient=pca)))

        total_count = 0
        ingredients_count = 0.0
        print "4"
        for ingredient in ingredients:
            counted = generated_text.count(ingredient)
            total_count += counted
            if counted > 0:
                ingredients_count += 1.0

        if len(ingredients) == 0:
            count_percentage = 0.0;
        else:
            count_percentage = (ingredients_count/len(ingredients)) * 100

        print "----START----"
        print str(count_percentage) + "% of the ingredients occured " + str(total_count) + " times in the text -----"
        print "Ingredients: " + str(ingredients) 
        print "Text: " + generated_text
        print "----END----"

        return {"text": generated_text, "ingredients": ingredients, "count": total_count, "ingredients_count": ingredients_count, "percentage": count_percentage}



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
        if config.use_word2vec:
            lstm_model = Word2VecLSTM(config)
        else:
            lstm_model = EmbeddingLSTM(config)

        # if config.generate:
        # This instructs gen_model to reuse the same variables as the model above
        scope.reuse_variables()
        if config.use_word2vec:
            gen_model = Word2VecLSTM(gen_config)
        else:
            gen_model = EmbeddingLSTM(gen_config)

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

    results = []

    total = {"count": 0, "percentage": 0.0, "generated": 0.0}
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with open(r'ingredient_inclusion.csv','a') as f:
        writer = csv.writer(f)

        # Open session and generate text
        with tf.Session() as session: 
            saver.restore(session, generator.store_location) 
            for batch in encoded_test: 
                for recipe in batch.get_all_recipes():
                    start_text = vocab.decode(recipe.get_sequence_i(0)[0][0])
                    result = generate_from(generator, start_text, recipe.get_multihot(), recipe.get_pca(), gen_model, ingredient_list, session)
                    total["count"] += result["count"]
                    total["generated"] += 1.0
                    total["percentage"] += result["percentage"]
                    results.append(result)
                    writer.writerow([result["count"], result["ingredients_count"], len(result["ingredients"]), result["percentage"]])
                    print "++++++ Average percentage: " + str(total["percentage"]/total["generated"]) +\
                            " over " + str(total["generated"]) + "+++++++"

    results.append(total)

    

if __name__=="__main__":
    main()