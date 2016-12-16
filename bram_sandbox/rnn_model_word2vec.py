from rnn_model_abtract import RNNLM_Model
import sys
import numpy as np
import tensorflow as tf

class Word2VecLSTM(RNNLM_Model):

    def __init__(self, config):
        RNNLM_Model.__init__(self, config)

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
                        self.initial_cell_state: np.zeros((self.config.batch_size, self.config.hidden_size)) if sequence_step == 0 else cell_state, #TODO
                        self.initial_hidden_state: np.zeros((self.config.batch_size, self.config.hidden_size)) if sequence_step == 0 else hidden_state,
                        self.dropout_placeholder: dp}
                loss, hidden_state, cell_state, _ = session.run(
                    [self.calculate_loss, self.final_hidden_state, self.final_cell_state, train_op], feed_dict=feed)
                total_loss.append(loss)
            if verbose and batch_step % verbose == 0:
                sys.stdout.write(
                    '\r{} / {} : pp = {}'.format(batch_step, len(recipe_batch_list), np.exp(np.mean(total_loss))))
                sys.stdout.flush()

            if verbose:
                sys.stdout.write('\r')

        return np.exp(np.mean(total_loss))

