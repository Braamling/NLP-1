from copy import deepcopy

import tensorflow as tf

from rnn_model_word2vec import Word2VecLSTM
from rnn_model_embedding import EmbeddingLSTM
from generator import Text_Generator

import json
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
    early_stopping = 4
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


def main():
    # Create config for the training and generator models
    config = Config()
    gen_config = deepcopy(config)

    # Set the batch size and step size of the generator model to 1
    gen_config.batch_size = gen_config.num_steps = 1


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

    # Train the RNN model if the is set in the config
    if config.train:
        lstm_model.train()

    # Generate text based on the passed word.
    if config.generate:
        generator = Text_Generator(gen_config, gen_model)

        start_text = "preheat"
        ingredients = ["chicken", "wrap", "tomato", "cheese", "pizza"]
        generator.generate_from(start_text, ingredients, gen_model.vocab)


if __name__=="__main__":
    main()