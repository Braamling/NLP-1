import sys
import time

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_dataset, Vocab
from utils import sample, get_random_multihot, get_ingredient_list_size

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss

class Text_Generator():

    def __init__(self, config, gen_model):
        self.config = config
        self.model = gen_model

    def generate_from(self, starting_text):
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        # Open session and generate text
        with tf.Session() as session:  

            # Retrieved configured session.         
            saver.restore(session, self.config.session_name)

            while starting_text:
                print ' '.join(self.generate_sentence(
                        session, starting_text=starting_text, temp=1.0))
                starting_text = raw_input('> ')


    def generate_text(self, session, starting_text='<endofrecipe>',
                      stop_length=100, stop_tokens=None, temp=1.0):
        """Generate text from the model.
        Args:
            session: tf.Session() object
            model: Object of type RNNLM_Model
            config: A Config() object
            starting_text: Initial text passed to model.
        Returns:
            output: List of word idxs
        """
        state = self.model.initial_cell_state.eval()


        # Imagine tokens as a batch size of one, length of len(tokens[0])
        tokens = [self.model.vocab.encode(word) for word in starting_text.split()]
        #pad_token = self.model.vocab.word_to_index[self.model.vocab.unknown]
        #inputs = [tokens[-config.num_steps:]] if len(tokens)>config.num_steps else [(config.num_steps-len(tokens))*[pad_token]+tokens]
        num = self.config.num_steps

        ing_list_size = get_ingredient_list_size(self.config.ingredients_data)
        # print('inputs:',inputs,[self.model.vocab.decode(widx) for widx in inputs[0]])
        
        for i in xrange(stop_length):
            inputs = [tokens[-num:]]
            feed_dict = {
                self.model.rnn_input_placeholder : inputs,
                self.model.ingredient_placeholder: np.array(get_random_multihot(ing_list_size, self.model.vocab)),
                self.model.dropout_placeholder : self.config.dropout,
                self.model.initial_cell_state : state
            }

            state, y_pred = session.run([self.model.final_cell_state, self.model.predictions[-1]], feed_dict = feed_dict)
            #print y_pred.shape # (1, len(vocab)), so the shape of y_pred[0] is (len(vocab),)
            next_word_idx = sample(y_pred[0], temperature=temp)
            tokens.append(next_word_idx)
            if stop_tokens and self.model.vocab.decode(tokens[-1]) in stop_tokens:
                break

        print('i:',i)
        output = [self.model.vocab.decode(word_idx) for word_idx in tokens]
        return output

    def generate_sentence(self, session, *args, **kwargs):
        """Convenice to generate a sentence from the model."""
        return self.generate_text(session, *args, stop_tokens=['<endofrecipe>'], **kwargs)

