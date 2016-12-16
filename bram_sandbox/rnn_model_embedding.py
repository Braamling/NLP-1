from rnn_model_abtract import RNNLM_Model
import sys
import numpy as np
from utils import get_ingredient_list_size
import tensorflow as tf


class EmbeddingLSTM(RNNLM_Model):

    def __init__(self):
        RNNLM_Model.__init__(self)

    def add_additional_placeholders(self):
        self.ingredient_placeholder = tf.placeholder(tf.float32,
                                                [None, get_ingredient_list_size(self.config.ingredients_data)],
                                                name="Ingredient_Input")
    def get_initial_cell_state(self):
        return self.add_ingredient_nn()

    def add_ingredient_nn(self):
        with tf.variable_scope('ingredient_nn') as scope:
            # TODO check these multiplications
            W1 = tf.get_variable('ing_W1', [get_ingredient_list_size(self.config.ingredients_data),
                                            self.config.ingredient_hidden_size])  # TODO get length of ingredient vector
            b1 = tf.get_variable('ing_b1', [self.config.ingredient_hidden_size])
            W2 = tf.get_variable('ing_W2', [self.config.ingredient_hidden_size,
                                            self.config.hidden_size])  # hidden_size is size of cell state
            b2 = tf.get_variable('ing_b2', [self.config.hidden_size])  # hidden_size is size of cell state
            hidden_layer = tf.nn.tanh(tf.matmul(self.ingredient_placeholder, W1) + b1)
            output_ingredient_nn = tf.nn.tanh(tf.matmul(hidden_layer, W2) + b2)
            return output_ingredient_nn

    def run_epoch(self, session, recipe_batch_list, train_op=None, verbose=1):

        total_loss = []
        for batch_step, recipe_batch in enumerate(recipe_batch_list):

            # hidden_state = self.initial_hidden_state.eval()
            dp = self.config.dropout
            if not train_op:
                train_op = tf.no_op()
                dp = 1

            for sequence_step in range(recipe_batch.get_max_sequence_size()):
                rnn_input_x, rnn_input_y = recipe_batch.get_all_sequence_i(sequence_step)
                feed = {self.rnn_input_placeholder: np.array(rnn_input_x),
                        self.rnn_labels_placeholder: np.array(rnn_input_y),
                        self.ingredient_placeholder: np.array(recipe_batch.get_all_multihots()),
                        self.initial_hidden_state: np.zeros(
                            (self.config.batch_size, self.config.hidden_size)) if sequence_step == 0 else hidden_state,
                        # TODO try for understanding tf.zeros([self.config.batch_size, self.config.hidden_size])
                        self.dropout_placeholder: dp}
                loss, hidden_state, _ = session.run(
                    [self.calculate_loss, self.final_hidden_state, train_op], feed_dict=feed)
                total_loss.append(loss)
            if verbose and batch_step % verbose == 0:
                sys.stdout.write(
                    '\r{} / {} : pp = {}'.format(batch_step, len(recipe_batch_list), np.exp(np.mean(total_loss))))
                sys.stdout.flush()

            if verbose:
                sys.stdout.write('\r')

        return np.exp(np.mean(total_loss))