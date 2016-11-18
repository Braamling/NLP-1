import numpy as np
# np.set_printoptions(threshold=np.nan)

batch_size = 2
n_steps = 50

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


with open("sec02-21.gold.tagged", 'r') as f:
    char_list = list(f.read())
batches = get_batches(char_list)
x, y = batches.next()

char_dict = get_onehot_dic(char_list)
onehot_dict = get_char_dic(char_dict)

#print(np.where(x[0] == 1)[0][0])
print [ onehot_dict[np.where(char == 1)[0][0]]for char in x]
print [ onehot_dict[np.where(char == 1)[0][0]]for char in y]