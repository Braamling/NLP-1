from copy import deepcopy

import tensorflow as tf

from rnn_model import RNNLM_Model
from generator import Text_Generator

"""
Please set the 

"""


class Config(object):
    # Train and/or generate the model when running the script
    train = False
    generate = True

    # Learning parameters
    batch_size = 10
    embed_size = 5
    hidden_size = 50
    num_steps = 40
    max_epochs = 10
    early_stopping = 2
    dropout = 0.9
    lr = 0.005

    # Training data locations
    encoded_train = 'data/recipes/1000.mmf'
    encoded_valid = 'data/recipes/2000.mmf'
    encoded_test = 'data/recipes/3000.mmf'

    # All recipe data.
    merged_data = 'data/recipes/merged.txt'

    # Session location and name for storing and loading stored model
    session_name = 'ptb_rnnlm.weights'
    store_location = './ptb_rnnlm.weights'


def main():
    # Create config for the training and generator models
    config = Config()
    gen_config = deepcopy(config)

    # Set the batch size and step size of the generator model to 1
    gen_config.batch_size = gen_config.num_steps = 1

    # We create the training model and generative model
    with tf.variable_scope('RNNLM') as scope:
          rnn_model = RNNLM_Model(config)

          # This instructs gen_model to reuse the same variables as the model above
          scope.reuse_variables()
          gen_model = RNNLM_Model(gen_config)

    # Train the RNN model if the is set in the config
    if config.train:
        rnn_model.train()

    # Generate text based on the passed word.
    if config.generate:
        generator = Text_Generator(gen_config, gen_model)

        generator.generate_from("preheat")


if __name__=="__main__":
    main()