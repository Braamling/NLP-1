import sys
import time

import os
import numpy as np

from utils import calculate_perplexity, get_dataset, Vocab, load_pickle_to_dict
from utils import sample, get_words_from_dataset
from utils import get_ingredient_list_size
import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss


class RNNLM_Model():

    def __init__(self, config ):
        self.config = config
        
        self.load_data(debug=False)
        self.add_placeholders()

        self.initial_cell_state = tf.zeros([self.config.batch_size, self.config.hidden_size])

        rnn_inputs = self.add_embedding()
        self.initial_hidden_state = self.add_ingredient_nn()
        rnn_outputs = self.add_rnn_model(rnn_inputs)
        self.outputs = self.add_projection(rnn_outputs)

        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
        output = tf.reshape(tf.concat(1, self.outputs), [-1,len(self.vocab)])
        self.calculate_loss = self.add_loss_op(output)
        self.train_step = self.add_training_op(self.calculate_loss)

    def load_data(self, debug=False):
        """Loads starter word-vectors and train/dev/test data. """
        self.vocab = Vocab()
        self.vocab.construct(get_words_from_dataset(self.config.merged_data))

        self.encoded_train = [recipe for recipe in\
                              get_dataset(self.config.encoded_train, 
                                          self.config.ingredients_data,
                                          self.vocab,
                                          self.config.num_steps,
                                          self.config.batch_size)]

        self.encoded_valid = [recipe for recipe in\
                              get_dataset(self.config.encoded_valid,
                                          self.config.ingredients_data,
                                          self.vocab,
                                          self.config.num_steps,
                                          self.config.batch_size)]

        self.encoded_test = [recipe for recipe in\
                              get_dataset(self.config.encoded_test,
                                          self.config.ingredients_data,
                                          self.vocab,
                                          self.config.num_steps,
                                          self.config.batch_size)]

        if debug:
            num_debug = 1024*3
            self.encoded_train = self.encoded_train[:num_debug]
            self.encoded_valid = self.encoded_valid[:num_debug]
            self.encoded_test = self.encoded_test[:num_debug]

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building
        code and will be fed data during training.  Note that when "None" is in a
        placeholder's shape, it's flexible

        """
        self.rnn_input_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='Input')
        self.rnn_labels_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='Target')
        self.ingredient_placeholder = tf.placeholder(tf.float32, [None, get_ingredient_list_size(self.config.ingredients_data)], name="Ingredient_Input") #TODO get length of ingredient vector
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')

    def add_ingredient_nn(self):
        with tf.variable_scope('ingredient_nn') as scope:
            # TODO check these multiplications
            W1 = tf.get_variable('ing_W1', [get_ingredient_list_size(self.config.ingredients_data), self.config.ingredient_hidden_size]) #TODO get length of ingredient vector
            b1 = tf.get_variable('ing_b1', [self.config.ingredient_hidden_size])
            W2 = tf.get_variable('ing_W2', [self.config.ingredient_hidden_size, self.config.hidden_size]) #hidden_size is size of cell state
            b2 = tf.get_variable('ing_b2', [self.config.hidden_size]) #hidden_size is size of cell state
            hidden_layer = tf.nn.tanh(tf.matmul(self.ingredient_placeholder, W1) + b1)
            output_ingredient_nn = tf.nn.tanh(tf.matmul(hidden_layer, W2) + b2)
            return output_ingredient_nn


    def add_embedding(self):
        """Add embedding layer.
    Returns:
            inputs: List of length num_steps, each of whose elements should be
                            a tensor of shape (batch_size, embed_size).
        """
        # The embedding lookup is currently only implemented for the CPU
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('Embedding', [len(self.vocab), self.config.embed_size])
            inputs = tf.nn.embedding_lookup(embedding, self.rnn_input_placeholder) # (batch_size, num_steps, embed_size)
            inputs = [tf.squeeze(x,[1]) for x in tf.split(1, self.config.num_steps, inputs)] # Each element is (batch_size, embed_size).
            return inputs

    def add_projection(self, rnn_outputs):
        """Adds a projection layer.
        Args:
            rnn_outputs: List of length num_steps, each of whose elements should be
                                     a tensor of shape (batch_size, hidden_size).
        Returns:
            outputs: List of length num_steps, each a tensor of shape
                             (batch_size, len(vocab))
        """
        with tf.variable_scope('Softmax') as scope:
            U = tf.get_variable('U', [self.config.hidden_size, len(self.vocab)])
            b_2 = tf.get_variable('b_2', [len(self.vocab)])
            outputs = [tf.matmul(rnn_output, U) + b_2 for rnn_output in rnn_outputs] # Each  rnn_output is a hidden layer states
        return outputs

    def add_loss_op(self, output):
        """Adds loss ops to the computational graph.
        Hint: Use tensorflow.python.ops.seq2seq.sequence_loss to implement sequence loss.
        Args:
            output: A tensor of shape (None, self.vocab)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        all_ones_weights = [tf.ones([self.config.batch_size * self.config.num_steps])]
        # output is logits
        loss = sequence_loss([output], \
                             [tf.reshape(self.rnn_labels_placeholder, [-1])], \
                             all_ones_weights) # , len(self.vocab)
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        tf.scalar_summary("cost", loss)
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = opt.minimize(loss, global_step=global_step)
        return train_op

    def add_rnn_model(self, rnn_inputs):
        """Creates the RNN LM model.

        Args:
            rnn_inputs: List of length num_steps, each of whose elements should be
                            a tensor of shape (batch_size, embed_size).
        Returns:
            outputs: List of length num_steps, each of whose elements should be
                             a tensor of shape (batch_size, hidden_size)
        """
        with tf.variable_scope('RNN') as scope:
            hidden_state = self.initial_hidden_state
            cell_state = self.initial_cell_state
            rnn_outputs = []
            for tstep, rnn_input in enumerate(rnn_inputs):
                if tstep > 0:
                    scope.reuse_variables()
                rnn_input = tf.nn.dropout(rnn_input, self.dropout_placeholder)
                lstm_input = tf.concat(1, [rnn_input, hidden_state]) # Possibly use output instead of hidden_state?

                H = tf.get_variable('H', [self.config.hidden_size + self.config.embed_size, self.config.hidden_size * 4]) # Wf
                b = tf.get_variable('b', [self.config.hidden_size * 4])
                f, i, j, o = tf.split(1, 4, tf.matmul(lstm_input, H) + b)

                # TODO Add dropout possibly
                forget_g = tf.nn.sigmoid(f)
                input_g = tf.nn.sigmoid(i)
                input_j_g = tf.nn.tanh(j)
                output_g = tf.nn.sigmoid(o)

                cell_state = tf.mul(cell_state, forget_g) + tf.mul(input_j_g, input_g)
                hidden_state = tf.mul(tf.tanh(cell_state), output_g)
                output = tf.nn.dropout(hidden_state, self.dropout_placeholder)
                rnn_outputs.append(output)

        self.final_cell_state = cell_state

        return rnn_outputs


    def train(self):
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as session:
            best_val_pp = float('inf')
            best_val_epoch = 0

            session.run(init)

            if os.path.exists(self.config.session_name):
                saver.restore(session, self.config.session_name)
            
            for epoch in xrange(self.config.max_epochs):
                print 'Epoch {}'.format(epoch)

                start = time.time()
                
                train_pp = self.run_epoch(session, self.encoded_train, train_op=self.train_step)
                valid_pp = self.run_epoch(session, self.encoded_valid)

                print 'Training perplexity: {}'.format(train_pp)
                print 'Validation perplexity: {}'.format(valid_pp)

                # Save the results if the results improved
                if valid_pp < best_val_pp:
                    best_val_pp = valid_pp
                    best_val_epoch = epoch
                    saver.save(session, self.config.store_location)

                # Stop if the early the prefered threshold has been achieved. 
                if epoch - best_val_epoch > self.config.early_stopping:
                    break

                print 'Total time: {}'.format(time.time() - start)

    def run_epoch(self, session, recipe_batch_list, train_op=None, verbose=10):

        total_loss = []
        for recipe_batch in recipe_batch_list:

            cell_state = self.initial_cell_state.eval()
            dp = self.config.dropout
            if not train_op:
                train_op = tf.no_op()
                dp = 1

            for batch_step in range(recipe_batch.get_max_sequence_size()):
                rnn_input_x, rnn_input_y = recipe_batch.get_all_sequence_i(batch_step)
                feed = {self.rnn_input_placeholder: np.array(rnn_input_x),
                        self.rnn_labels_placeholder: np.array(rnn_input_y),
                        self.ingredient_placeholder: np.array(recipe_batch.get_all_multihots()),
                        self.initial_cell_state: np.zeros((self.config.batch_size, self.config.hidden_size)) if batch_step == 0 else cell_state, #TODO try for understanding tf.zeros([self.config.batch_size, self.config.hidden_size])
                        self.dropout_placeholder: dp}
                loss, cell_state, _ = session.run(
                            [self.calculate_loss, self.final_cell_state, train_op], feed_dict=feed)
                total_loss.append(loss)
                if verbose and batch_step % verbose == 0:
                   sys.stdout.write('\r{} / {} : pp = {}'.format(batch_step, recipe_batch.get_max_sequence_size(), np.exp(np.mean(total_loss))))
                   sys.stdout.flush()

            if verbose:
                sys.stdout.write('\r')

        return np.exp(np.mean(total_loss))

