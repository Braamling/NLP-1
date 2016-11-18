import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell

# Parameters
learning_rate = 0.001
epochs = 4
batch_size = 1
display_step = 1000
model_dir_name = 'testModel/'
save_steps = 1
training = True


# Network Parameters
n_input = 83 # alfabet size for one hot
n_steps = 20
n_hidden = 10
n_output = n_input

# tf Graph
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_steps, n_output])

# Define weights
weights = tf.Variable(tf.random_normal([n_hidden, n_output]))
biases = tf.Variable(tf.random_normal([n_output]))

def get_onehot_dic(char_list):
    # returns a dictionary that represents character in one hot representation
    char_set = set(char_list)
    char_len = len(char_set)
    char_dict ={}
    for i, char in enumerate(char_set):
        char_vec = np.zeros(char_len)
        char_vec[i] = 1
        char_dict.update({char: char_vec})

    return char_dict

def get_char_dic(char_dict):
    onehot_dic = {}

    for i, (key, value) in enumerate(char_dict.items()):
        onehot_dic.update({i: key})

    return onehot_dic

def get_batches(char_list):

    char_dict = get_onehot_dic(char_list)
    # onehot_dic = get_char_dic(char_dict)

    data_len = len(char_list)
    n = int((float(data_len) / (n_steps * batch_size)))
    for i in range(n):

        current_i = i * n_steps * batch_size

        if current_i + (n_steps * batch_size) <= data_len-1:
            # return batches of size batch_size
            batch_x = []
            batch_y = []
            for _ in range(batch_size):
                batch_x.append([char_dict[char] for char in
                           char_list[current_i: current_i + n_steps]])
                batch_y.append([char_dict[char] for char in
                                char_list[current_i + 1: current_i + n_steps + 1]])


        # else:
        #     # return last batch of size <batch_size if it exists
        #     batch_x = [char_dict[char] for char in
        #              char_list[current_i: data_len]]
        #     batch_y = batch_x[1:]
        #     batch_y.append(char_dict[char_list[0]])

        yield np.asarray(batch_x), np.asarray(batch_y)

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)
    # lstm cell
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias = 1.0)

    # Get lstm output
    output, states = rnn.rnn(lstm_cell, x, dtype = tf.float32)

    # Output for the next batch
    return [tf.matmul(output_x, weights) + biases for output_x in output]

pred = RNN(x, weights, biases)

# cost function and optimizer(to speed things up)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initialize variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state("my_path_to_model")
    with open("sec02-21.gold.tagged", 'r') as f:
        char_list = list(f.read())
    n = int((float(len(char_list)) / n_steps))
    # Keep training until reach max iterations

    ckpt = tf.train.get_checkpoint_state(model_dir_name)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print "No checkpoint found!"
    if training:
        for i in range(epochs):
            # Shuffle?
            text_batch = get_batches(char_list)

            for j in range(n):
                batch_x, batch_y = text_batch.next()
                sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
                if j % display_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

                    print("Iter " + str(j*batch_size) + ", Minibatch Loss=" + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
                    saver.save(sess, model_dir_name + 'model.ckpt')
                    print("Model saved in file: %s" % model_dir_name + 'model.ckpt')
    else:
        text_batch = get_batches(char_list)
        batch_x, batch_y = text_batch.next()

        char_dict = get_onehot_dic(char_list)
        onehot_dict = get_char_dic(char_dict)

        stuff = sess.run(pred, feed_dict={x: batch_x, y:batch_y})

        sentence = []
        sentence2 = []
        for vec in batch_x[0]:
            character = onehot_dict[np.argmax(vec)]
            sentence.append(character)
        for vec in stuff:
            character = onehot_dict[np.argmax(vec)]
            sentence2.append(character)

        print ' '.join(sentence)
        print ' '.join(sentence2)


    print("Optimization Finished")

