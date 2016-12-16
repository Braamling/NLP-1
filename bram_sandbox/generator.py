import sys
import time

import numpy as np
import json
from copy import deepcopy

from utils import calculate_perplexity, get_dataset, Vocab, get_multi_hot
from utils import sample, get_random_multihot, load_pickle_to_dict

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss

class Text_Generator():

    def __init__(self, config, gen_model):
        self.config = config
        self.model = gen_model

        with open('config.json') as data_file:    
            config = json.load(data_file)

        self.session_name = config['session_name']
        self.store_location = config['store_location']



    def generate_from(self, starting_text, ingredients, vocab):
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()


        # Open session and generate text
        with tf.Session() as session: 


            # Retrieved configured session.         
            saver.restore(session, self.store_location)

            while starting_text:
                ingredients = vocab.encode_list(ingredients)

                print ' '.join(self.generate_sentence(
                        session, starting_text=starting_text, 
                        ingredients=ingredients, temp=1.0))
                starting_text = raw_input('start text> ')
                ingredients = raw_input('ingredents (, )> ')
                ingredients = ingredients.split(', ')


    def generate_text(self, session, starting_text ='<endofrecipe>',
                      ingredients = ["<unk>"], stop_length = 100, stop_tokens = None, temp = 1.0, multi_hot = None, pca_ingredient=None):
        """Generate text from the model.
        Args:
            session: tf.Session() object
            model: Object of type RNNLM_Model
            config: A Config() object
            starting_text: Initial text passed to model.
        Returns:
            output: List of word idxs
        """
        hidden_state = self.model.initial_cell_state
        cell_state = self.model.initial_hidden_state

        # Imagine tokens as a batch size of one, length of len(tokens[0])
        tokens = [self.model.vocab.encode(word) for word in starting_text.split()]
        #pad_token = self.model.vocab.word_to_index[self.model.vocab.unknown]
        #inputs = [tokens[-config.num_steps:]] if len(tokens)>config.num_steps else [(config.num_steps-len(tokens))*[pad_token]+tokens]
        num = self.config.num_steps

        if multi_hot is None:
            ing_list = load_pickle_to_dict(self.config.ingredients_data)
            ingredients = get_multi_hot(ingredients, ing_list)
            ingredients = ingredients.reshape(1, len(ingredients))
        else:
            ingredients = multi_hot.reshape(1, len(multi_hot))
        # print('inputs:',inputs,[self.model.vocab.decode(widx) for widx in inputs[0]])
        
        for i in xrange(stop_length):
            inputs = [tokens[-num:]]

            if self.config.use_word2vec:
                feed_dict = {
                    self.model.rnn_input_placeholder : inputs,
                    self.model.dropout_placeholder : self.config.dropout,
                    self.model.initial_cell_state: pca_ingredient,
                    self.model.initial_hidden_state: hidden_state,
                    self.model.initial_cell_state : cell_state
                }
                cell_state, hidden_state, y_pred = session.run([self.model.final_cell_state, self.model.final_hidden_state, self.model.predictions[-1]], feed_dict = feed_dict)
            else:
                feed_dict = {
                    self.model.rnn_input_placeholder : inputs,
                    self.model.ingredient_placeholder: ingredients,
                    self.model.dropout_placeholder : self.config.dropout,
                    self.model.initial_cell_state : state
                }
                state, y_pred = session.run([self.model.final_cell_state, self.model.predictions[-1]], feed_dict = feed_dict)

            next_word_idx = sample(y_pred[0], temperature=temp)
            tokens.append(next_word_idx)
            if stop_tokens and self.model.vocab.decode(tokens[-1]) in stop_tokens:
                break

        output = [self.model.vocab.decode(word_idx) for word_idx in tokens]
        return output

    def generate_sentence(self, session, *args, **kwargs):
        """Convenice to generate a sentence from the model."""
        return self.generate_text(session, *args, stop_tokens=['<endofrecipe>'], **kwargs)

