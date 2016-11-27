import sys
import time

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_dataset, Vocab
from utils import ptb_iterator, sample

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss


class Config(object):
  batch_size = 10
  embed_size = 5
  hidden_size = 50
  num_steps = 40
  max_epochs = 10
  early_stopping = 2
  dropout = 0.9
  lr = 0.005

class RNNLM_Model():
    def load_data(self, debug=False):
        """Loads starter word-vectors and train/dev/test data. """
        self.vocab = Vocab()
        self.vocab.construct(get_dataset('data/recipes/merged.txt'))
        self.encoded_train = np.array(
            [self.vocab.encode(word) for word in get_dataset('data/recipes/1000.mmf')],
            dtype=np.int32)
        self.encoded_valid = np.array(
            [self.vocab.encode(word) for word in get_dataset('data/recipes/2000.mmf')],
            dtype=np.int32)
        self.encoded_test = np.array(
            [self.vocab.encode(word) for word in get_dataset('data/recipes/3000.mmf')],
            dtype=np.int32)
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
      self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='Input')
      self.labels_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps], name='Target')
      self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')

    def add_embedding(self):
      """Add embedding layer.
    Returns:
        inputs: List of length num_steps, each of whose elements should be
                a tensor of shape (batch_size, embed_size).
      """
      # The embedding lookup is currently only implemented for the CPU
      with tf.device('/cpu:0'):
        embedding = tf.get_variable('Embedding', [len(self.vocab), self.config.embed_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder) # (data_size, num_steps, embed_size)
        inputs = [tf.squeeze(x,[1]) for x in tf.split(1, self.config.num_steps, inputs)] # Each element is (data_size, embed_size).
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
          [tf.reshape(self.labels_placeholder, [-1])],\
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
      train_op = opt.minimize(loss,global_step=global_step)
      return train_op

    def __init__(self, config):
        self.config = config
        
        self.load_data(debug=False)
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.rnn_outputs = self.add_model(self.inputs)
        self.outputs = self.add_projection(self.rnn_outputs)

        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
        output = tf.reshape(tf.concat(1, self.outputs), [-1,len(self.vocab)])
        self.calculate_loss = self.add_loss_op(output)
        self.train_step = self.add_training_op(self.calculate_loss)


    def add_model(self, inputs):
        """Creates the RNN LM model.

        Args:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size)
        """
        with tf.variable_scope('RNN') as scope:
            self.initial_state = tf.zeros([self.config.batch_size, self.config.hidden_size])
            hidden_state = self.initial_state
            rnn_outputs = []
            for tstep,rnn_input in enumerate(inputs):
                if tstep > 0: scope.reuse_variables()
                H = tf.get_variable('H', [self.config.hidden_size, self.config.hidden_size]) # Wh
                I = tf.get_variable('I', [self.config.embed_size, self.config.hidden_size]) # Wx
                b1 = tf.get_variable('b1', [self.config.hidden_size])
                rnn_input = tf.nn.dropout(rnn_input, self.dropout_placeholder)
                hidden_state = tf.nn.tanh( tf.matmul( rnn_input, I) + b1) + tf.nn.tanh(tf.matmul(hidden_state, H))
                output = tf.nn.dropout(hidden_state, self.dropout_placeholder)
                rnn_outputs.append(output)

        self.final_state = rnn_outputs[-1]

        return rnn_outputs


    def run_epoch(self, session, data, train_op=None, verbose=10):
      config = self.config
      dp = config.dropout
      if not train_op:
        train_op = tf.no_op()
        dp = 1

      total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))

      total_loss = []
      state = self.initial_state.eval()
      for step, (x, y) in enumerate(
        ptb_iterator(data, config.batch_size, config.num_steps)):
        # We need to pass in the initial state and retrieve the final state to give
        # the RNN proper history
        feed = {self.input_placeholder: x,
                self.labels_placeholder: y,
                self.initial_state: state,
                self.dropout_placeholder: dp}
        loss, state, _ = session.run(
            [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
        total_loss.append(loss)
        if verbose and step % verbose == 0:
            sys.stdout.write('\r{} / {} : pp = {}'.format(
                step, total_steps, np.exp(np.mean(total_loss))))
            sys.stdout.flush()
      if verbose:
        sys.stdout.write('\r')
      return np.exp(np.mean(total_loss))

def generate_text(session, model, config, starting_text='<eos>',
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
  state = model.initial_state.eval()
  # Imagine tokens as a batch size of one, length of len(tokens[0])
  tokens = [model.vocab.encode(word) for word in starting_text.split()]
  #pad_token = model.vocab.word_to_index[model.vocab.unknown]
  #inputs = [tokens[-config.num_steps:]] if len(tokens)>config.num_steps else [(config.num_steps-len(tokens))*[pad_token]+tokens]
  num = config.num_steps
  print('num:',num)
  inputs = [tokens[-num:]]
  print('inputs:',inputs,[model.vocab.decode(widx) for widx in inputs[0]])
  for i in xrange(stop_length):
    feed_dict = {
      model.input_placeholder : inputs,
      model.dropout_placeholder : config.dropout,
      model.initial_state : state
    }
    state, y_pred = session.run([model.final_state, model.predictions[-1]], feed_dict = feed_dict)
    #print y_pred.shape # (1, len(vocab)), so the shape of y_pred[0] is (len(vocab),)
    next_word_idx = sample(y_pred[0], temperature=temp)
    tokens.append(next_word_idx)
    if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
      break
  print('i:',i)
  output = [model.vocab.decode(word_idx) for word_idx in tokens]
  return output

def generate_sentence(session, model, config, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
  config = Config()
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1

  # We create the training model and generative model
  with tf.variable_scope('RNNLM') as scope:
    model = RNNLM_Model(config)
    # This instructs gen_model to reuse the same variables as the model above
    scope.reuse_variables()
    gen_model = RNNLM_Model(gen_config)

  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  with tf.Session() as session:
    best_val_pp = float('inf')
    best_val_epoch = 0
  
    session.run(init)
    
    saver.restore(session, 'ptb_rnnlm.weights')
    
    # for epoch in xrange(config.max_epochs):
    #   print 'Epoch {}'.format(epoch)
    #   start = time.time()
      
    #   train_pp = model.run_epoch(
    #       session, model.encoded_train,
    #       train_op=model.train_step)
    #   valid_pp = model.run_epoch(session, model.encoded_valid)
    #   print 'Training perplexity: {}'.format(train_pp)
    #   print 'Validation perplexity: {}'.format(valid_pp)
    #   if valid_pp < best_val_pp:
    #     best_val_pp = valid_pp
    #     best_val_epoch = epoch
    #     saver.save(session, './ptb_rnnlm.weights')
    #   if epoch - best_val_epoch > config.early_stopping:
    #     break
    #   print 'Total time: {}'.format(time.time() - start)
    
    saver.restore(session, 'ptb_rnnlm.weights')
    #test_pp = model.run_epoch(session, model.encoded_test)
    # print '=-=' * 5
    # print 'Test perplexity: {}'.format(test_pp)
    # print '=-=' * 5
    starting_text = 'preheat'
    while starting_text:
      print ' '.join(generate_sentence(
          session, gen_model, gen_config, starting_text=starting_text, temp=1.0))
      starting_text = raw_input('> ')

if __name__=="__main__":
    test_RNNLM()
    