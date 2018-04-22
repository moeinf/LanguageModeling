##This script contains funtions for pre-processing text data,
# and generating synthetic data according to several models

import numpy as np
import scipy
import nltk
from random import shuffle

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def non_linear_map(x,percentile):
    th = np.percentile(x,percentile)
    x[x<th] = 0
    x =x / sum(x)
    return x

#########preprocessing functions for real-data

def preProcess(train_filename,validation_filename, test_filename, vocab_size,batch_size):
    unknown_token = 'UNKNOWN'
    f_read_train = open(train_filename,'r')
    f_read_validation = open(validation_filename,'r')
    f_read_test = open(test_filename,'r')
    # train_word_list=[]
    validation_word_list = []
    test_word_list=[]
    train_word_list = f_read_train.read().replace("\n", "<eos>").split()
    validation_word_list = f_read_validation.read().replace("\n", "<eos>").split()
    test_word_list = f_read_test.read().replace("\n", "<eos>").split()
    # for line in f_read_train:
    #     line = line.split()
    #     for word in line:
    #         train_word_list.append(word)
    # for line in f_read_validation:
    #     line = line.split()
    #     for word in line:
    #         validation_word_list.append(word)
    # for line in f_read_test:
    #     line = line.split()
    #     for word in line:
    #         test_word_list.append(word)

    print ("total number of words in train set:", len(train_word_list))
    print ("total number of words in validation set:", len(validation_word_list))
    print ("total number of words in test set:", len(test_word_list))

    # print ("first words in training set: ", train_word_list[120:150])

    fdist = nltk.FreqDist(train_word_list)
    print ("unique words:", len(fdist.keys()))
    vocab = fdist.most_common(vocab_size-1)
    print ("the least common word:", vocab[-1])
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    effective_train_length = len(train_word_list) // batch_size * batch_size
    effective_test_length = len(test_word_list) // batch_size * batch_size
    effective_validation_length = len(validation_word_list) // batch_size * batch_size

    train_word_list = train_word_list[:effective_train_length]
    test_word_list = test_word_list[:effective_test_length]
    validation_word_list = validation_word_list[:effective_validation_length]

    train_text_seq = [w if w in word_to_index else unknown_token for w in train_word_list]
    validation_text_seq = [w if w in word_to_index else unknown_token for w in validation_word_list]
    test_text_seq = [w if w in word_to_index else unknown_token for w in test_word_list]

    x_train = [word_to_index[w] for w in train_text_seq]
    x_train = np.array(x_train)
    x_validation = [word_to_index[w] for w in validation_text_seq]
    x_validation = np.array(x_validation)
    x_test = [word_to_index[w] for w in test_text_seq]
    x_test = np.array(x_test)

    y_train = np.roll(x_train,-1)
    y_train[-1] = 0
    y_validation = np.roll(x_validation, -1)
    y_validation[-1] = 0
    y_test = np.roll(x_test, -1)
    y_test[-1] = 0

    x_train = x_train.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y_train = y_train.reshape((batch_size, -1))

    x_validation = x_validation.reshape((batch_size, -1))
    y_validation = y_validation.reshape((batch_size,-1))

    x_test = x_test.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y_test = y_test.reshape((batch_size, -1))

    return (x_train,y_train,x_validation,y_validation,x_test,y_test,
            effective_train_length, effective_validation_length, effective_test_length)



def gatedPreProcess(train_filename, validation_filename, test_filename, vocab_size, batch_size):
    unknown_token = 'UNKNOWN'
    f_read_train = open(train_filename, 'r')
    f_read_validation = open(validation_filename, 'r')
    f_read_test = open(test_filename, 'r')
    train_word_list = []
    validation_word_list = []
    test_word_list = []
    for line in f_read_train:
        line = line.split()
        for word in line:
            train_word_list.append(word)
    for line in f_read_validation:
        line = line.split()
        for word in line:
            validation_word_list.append(word)
    for line in f_read_test:
        line = line.split()
        for word in line:
            test_word_list.append(word)

    print ("total number of words in train set:", len(train_word_list))
    print ("total number of words in validation set:", len(validation_word_list))
    print ("total number of words in test set:", len(test_word_list))

    fdist = nltk.FreqDist(train_word_list)
    print ("unique words:", len(fdist.keys()))
    vocab = fdist.most_common(vocab_size - 1)
    print ("the least common word:", vocab[-1])
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    effective_train_length = len(train_word_list) // batch_size * batch_size
    effective_test_length = len(test_word_list) // batch_size * batch_size
    effective_validation_length = len(validation_word_list) // batch_size * batch_size

    train_word_list = train_word_list[:effective_train_length]
    test_word_list = test_word_list[:effective_test_length]
    validation_word_list = validation_word_list[:effective_validation_length]

    train_text_seq = [w if w in word_to_index else unknown_token for w in train_word_list]
    validation_text_seq = [w if w in word_to_index else unknown_token for w in validation_word_list]
    test_text_seq = [w if w in word_to_index else unknown_token for w in test_word_list]

    x_train = [word_to_index[w] for w in train_text_seq]
    x_train = np.array(x_train)
    x_validation = [word_to_index[w] for w in validation_text_seq]
    x_validation = np.array(x_validation)
    x_test = [word_to_index[w] for w in test_text_seq]
    x_test = np.array(x_test)

    y_train = np.roll(x_train, -1)
    y_train[-1] = 0
    y_validation = np.roll(x_validation, -1)
    y_validation[-1] = 0
    y_test = np.roll(x_test, -1)
    y_test[-1] = 0

    train_data = [(x_train[i],y_train[i]) for i in range(len(x_train))]
    shuffle(train_data)
    validation_data = [(x_validation[i], y_validation[i]) for i in range(len(x_validation))]
    shuffle(validation_data)
    test_data = [(x_test[i], y_test[i]) for i in range(len(x_test))]
    shuffle(test_data)
    print ([(index_to_word[train_data[i][0]],index_to_word[train_data[i][1]]) for i in range(10)])

    return (train_data,test_data,validation_data,effective_train_length, effective_validation_length, effective_test_length)

####### preprocessing functions for synthetic data
def preProcessSynthetic(batch_size,data_filename):


    files = np.load(data_filename)
    x_train = files['x_train']
    x_test = files['x_test']
    x_valid = files['x_valid']
    entropy = files['entropy']
    print ("entropy is: ", entropy)

    effective_train_len = len(x_train) // batch_size * batch_size
    effective_test_len = len(x_test) // batch_size * batch_size
    effective_valid_len= len(x_valid) // batch_size * batch_size

    x_train = np.array(x_train[:effective_train_len])
    x_valid = np.array(x_valid[:effective_valid_len])
    x_test = np.array(x_test[:effective_test_len])

    y_train = np.roll(x_train, -1)
    y_train[-1] = 0
    y_valid = np.roll(x_valid, -1)
    y_valid[-1] = 0
    y_test = np.roll(x_test, -1)
    y_test[-1] = 0

    x_train = x_train.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y_train = y_train.reshape((batch_size, -1))

    x_valid = x_valid.reshape((batch_size, -1))
    y_valid = y_valid.reshape((batch_size, -1))

    x_test = x_test.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y_test = y_test.reshape((batch_size, -1))


    return (x_train, y_train, x_valid, y_valid, x_test, y_test,
            effective_train_len, effective_valid_len, effective_test_len)

##converts bigrams to sequence and then pre-process
def gatedPreProcessSyn(data_filename,batch_size):


    files = np.load(data_filename)
    x_train = files['x_train']
    x_test = files['x_test']
    x_valid = files['x_valid']

    effective_train_len = len(x_train) // batch_size * batch_size
    effective_test_len = len(x_test) // batch_size * batch_size
    effective_valid_len= len(x_valid) // batch_size * batch_size

    x_train = np.array(x_train[:effective_train_len])
    x_valid = np.array(x_valid[:effective_valid_len])
    x_test = np.array(x_test[:effective_test_len])

    y_train = np.roll(x_train, -1)
    y_train[-1] = 0
    y_valid = np.roll(x_valid, -1)
    y_valid[-1] = 0
    y_test = np.roll(x_test, -1)
    y_test[-1] = 0

    train_data = [(x_train[i], y_train[i]) for i in range(len(x_train))]
    shuffle(train_data)
    validation_data = [(x_valid[i], y_valid[i]) for i in range(len(x_valid))]
    shuffle(validation_data)
    test_data = [(x_test[i], y_test[i]) for i in range(len(x_test))]
    shuffle(test_data)

    return (train_data, test_data, validation_data,
    effective_train_len, effective_valid_len, effective_test_len)


######## data generation functions

def generate_syn_closed_hmm(vocab_size, state_size, train_seq_len,
                       test_seq_len, valid_seq_len):
    shift = int(0.2 * vocab_size)
    alpha = 1.6
    W = []
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(vocab_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
        row = sorted(row)
        row = [y for (x, y) in row]
        W.append(row)
    W = np.asarray(W)

    shift = int(0.2 * vocab_size)
    alpha = 1.8
    Hx = []
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(state_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(vocab_size):
        row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
        row = sorted(row)
        row = [y for (x, y) in row]
        Hx.append(row)
    Hx = np.asarray(Hx)

    shift = int(0.25 * vocab_size)
    alpha = 2
    Hh = []
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(state_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
        row = sorted(row)
        row = [y for (x, y) in row]
        Hh.append(row)
    Hh = np.asarray(Hh)


    # Hx = np.random.uniform(0,10,size = [vocab_size,state_size])
    # Hh =np.random.uniform(0,10,size=[state_size,state_size])

    # Hx = Hx /Hx.sum(axis=1,keepdims=True)
    # Hh = Hh / Hh.sum(axis=1,keepdims=True)

    beta = np.random.uniform(0,1)

    P_eq = np.multiply(beta, np.matmul(W,Hx)) + np.multiply((1-beta),Hh)

    ##finding the stationary distribution of h
    _, v = scipy.linalg.eig(P_eq, right=False, left=True)
    v = v.real
    st = v[:, 0] / sum(v[:, 0])
    print "sum of stationary distribution is: ", sum(st)

    ##finding the entropy of the source
    entropy = 0
    for i in range(state_size):
        ent = 0
        for j in range(vocab_size):
            if W[i, j] > 0:
                ent = ent + W[i, j] * np.log(W[i, j])
        ent = ent * st[i]
        entropy = entropy + ent
    entropy = -1 * entropy
    #
    # emp_entropy = 0

    # h_entropy = 0
    # for i in range(len(st)):
    #     if st[i] > 0:
    #         h_entropy = h_entropy + st[i] * np.log(st[i])
    # h_entropy = -1 * h_entropy

    # print "entropy of source is: ", entropy

    h_train_seq = [0] * train_seq_len
    x_train_seq = [0] * train_seq_len
    h_test_seq = [0] * test_seq_len
    x_test_seq = [0] * test_seq_len
    h_valid_seq = [0] * valid_seq_len
    x_valid_seq = [0] * valid_seq_len

    x_train_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[0])
    x_test_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[1])
    x_valid_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[2])

    h_train_seq[0] = np.random.choice(np.arange(0, state_size),p=Hx[x_train_seq[0]])
    h_valid_seq[0] = np.random.choice(np.arange(0, state_size), p=Hx[x_valid_seq[0]])
    h_test_seq[0] = np.random.choice(np.arange(0, state_size), p=Hx[x_test_seq[0]])

    for i in range(1,train_seq_len):
        idx = h_train_seq[i-1]
        samp_p = W[idx]
        x_train_seq[i] = np.random.choice(np.arange(0, vocab_size),
                                          p=samp_p)
        idx = x_train_seq[i]
        hidx = h_train_seq[i-1]
        h_train_seq[i] = np.random.choice(np.arange(0, state_size),
                                          p=(beta*Hx[idx]+(1-beta)*Hh[hidx]))


    for i in range(1,valid_seq_len):
        idx = h_valid_seq[i-1]
        x_valid_seq[i] = np.random.choice(np.arange(0, vocab_size),
                                          p=W[idx])
        idx = x_valid_seq[i]
        hidx = h_valid_seq[i-1]
        h_valid_seq[i] = np.random.choice(np.arange(0, state_size),
                                          p=(beta*Hx[idx]+(1-beta)*Hh[hidx]))

    for i in range(1,test_seq_len):
        idx = h_test_seq[i-1]
        x_test_seq[i] = np.random.choice(np.arange(0, vocab_size),
                                         p=W[idx])
        idx = x_test_seq[i]
        hidx = h_test_seq[i-1]
        h_test_seq[i] = np.random.choice(np.arange(0, state_size),
                                         p=(beta*Hx[idx] +(1-beta)*Hh[hidx]))

    entropy = 0
    h_vector = np.array([1.0 / state_size for _ in range(state_size)])
    for i in range(test_seq_len):
        x_prob = np.matmul(h_vector, W)
        sample = x_test_seq[i]
        if i > test_seq_len/2:
            entropy = entropy + np.log(x_prob[sample])
        h_vector = np.multiply((1-beta), np.matmul(h_vector, Hh)) +\
                   np.multiply(beta, Hx[sample])
    entropy = -2 * entropy / test_seq_len

    ent_lower = 0
    for i in range(test_seq_len/2,test_seq_len):
        x_prev = x_test_seq[i-1]
        h_prev = h_test_seq[i-1]
        h_prob = np.multiply(beta,Hx[x_prev]) + np.multiply((1-beta),Hh[h_prev])
        x_prob = np.matmul(h_prob,W)
        for e in x_prob:
            if e > 0:
                ent_lower = ent_lower + e*np.log(e)
    ent_lower = -2 * ent_lower / test_seq_len
    print ("upper bound on the entropy of test set is: ", entropy)
    print ("lower bound on the entropy of test set is: ", ent_lower)
    return W, Hx, Hh, beta, st, entropy, x_train_seq, x_valid_seq, x_test_seq



##generate a sequence according to the recurrent model
def generate_syn(vocab_size, state_size, train_seq_len,
                       test_seq_len, valid_seq_len):
    shift = int(0.2 * vocab_size)
    alpha = 1.3
    W = []
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(vocab_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
        row = sorted(row)
        row = [y for (x, y) in row]
        W.append(row)
    W = np.asarray(W)

    shift = int(0.15 * state_size)
    alpha = 1.8
    Hx = []
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(state_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(vocab_size):
        row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
        row = sorted(row)
        row = [y for (x, y) in row]
        Hx.append(row)
    Hx = np.asarray(Hx)

    shift = int(0.2 * state_size)
    alpha = 2
    Hh = []
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(state_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
        row = sorted(row)
        row = [y for (x, y) in row]
        Hh.append(row)
    Hh = np.asarray(Hh)


    beta = np.random.uniform(0,1)

    P_eq = np.multiply(beta, np.matmul(W,Hx)) + np.multiply((1-beta),Hh)

    ##finding the stationary distribution of h
    _, v = scipy.linalg.eig(P_eq, right=False, left=True)
    v = v.real
    st = v[:, 0] / sum(v[:, 0])
    print "sum of stationary distribution is: ", sum(st)

    ##finding the entropy of the source
    entropy = 0
    for i in range(state_size):
        ent = 0
        for j in range(vocab_size):
            if W[i, j] > 0:
                ent = ent + W[i, j] * np.log(W[i, j])
        ent = ent * st[i]
        entropy = entropy + ent
    entropy = -1 * entropy

    # emp_entropy = 0

    # h_entropy = 0
    # for i in range(len(st)):
    #     if st[i] > 0:
    #         h_entropy = h_entropy + st[i] * np.log(st[i])
    # h_entropy = -1 * h_entropy

    print "entropy of source is: ", entropy

    h_train_seq = [0] * train_seq_len
    x_train_seq = [0] * train_seq_len
    h_test_seq = [0] * test_seq_len
    x_test_seq = [0] * test_seq_len
    h_valid_seq = [0] * valid_seq_len
    x_valid_seq = [0] * valid_seq_len

    x_train_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[0])
    x_test_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[1])
    x_valid_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[2])

    h_train_seq[0] = np.random.choice(np.arange(0, state_size),p=Hx[x_train_seq[0]])
    h_valid_seq[0] = np.random.choice(np.arange(0, state_size), p=Hx[x_valid_seq[0]])
    h_test_seq[0] = np.random.choice(np.arange(0, state_size), p=Hx[x_test_seq[0]])

    for i in range(1,train_seq_len):
        idx = h_train_seq[i-1]
        samp_p = W[idx]
        # for j in range(vocab_size):
        #     if samp_p[j] > 0:
        #         emp_entropy = emp_entropy + samp_p[j] * np.log(samp_p[j])
        x_train_seq[i] = np.random.choice(np.arange(0, vocab_size), p=samp_p)
        idx = x_train_seq[i]
        hidx = h_train_seq[i-1]
        h_train_seq[i] = np.random.choice(np.arange(0, state_size), p=(beta*Hx[idx]+(1-beta)*Hh[hidx]))
    # emp_entropy = -1.0 * emp_entropy
    # emp_entropy = emp_entropy /train_seq_len
    # print ("empirical entropy is: ", emp_entropy)
    for i in range(1,valid_seq_len):
        idx = h_valid_seq[i-1]
        x_valid_seq[i] = np.random.choice(np.arange(0, vocab_size), p=W[idx])
        idx = x_valid_seq[i]
        hidx = h_valid_seq[i-1]
        h_valid_seq[i] = np.random.choice(np.arange(0, state_size), p=(beta*Hx[idx]+(1-beta)*Hh[hidx]))

    for i in range(1,test_seq_len):
        idx = h_test_seq[i-1]
        x_test_seq[i] = np.random.choice(np.arange(0, vocab_size), p=W[idx])
        idx = x_test_seq[i]
        hidx = h_test_seq[i-1]
        h_test_seq[i] = np.random.choice(np.arange(0, state_size), p=(beta*Hx[idx] +(1-beta)*Hh[hidx]))

    return W, Hx, Hh, beta, st, entropy, x_train_seq, x_valid_seq, x_test_seq

##generate a sequence according to the recurrent model, unique solution
def generate_syn_unique(vocab_size, state_size, train_seq_len,
                       test_seq_len, valid_seq_len):
    shift = int(0.2 * vocab_size)
    fixed_rows = int(state_size/2)
    alpha = 2
    W = np.identity(state_size)

    # W = []
    # cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(vocab_size)]
    # s = sum(y for (x, y) in cand)
    # cand = [(x, y / s) for (x, y) in cand]
    # for _ in range(state_size-fixed_rows):
    #     row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
    #     row = sorted(row)
    #     row = [y for (x, y) in row]
    #     W.append(row)
    # for i in range(fixed_rows):
    #     row = [0 for _ in range(vocab_size)]
    #     row[i] = 1
    #     W.append(row)
    # W = np.asarray(W)

    Hx=np.zeros([vocab_size,state_size])
    for i in range(len(Hx)-1):
        Hx[i,i+1] = 1
    Hx[-1,0]=1

    # shift = int(0.15 * state_size)
    # alpha = 1.8
    # Hx = []
    # cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(state_size)]
    # s = sum(y for (x, y) in cand)
    # cand = [(x, y / s) for (x, y) in cand]
    # for _ in range(vocab_size):
    #     row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
    #     row = sorted(row)
    #     row = [y for (x, y) in row]
    #     Hx.append(row)
    # Hx = np.asarray(Hx)

    shift = int(0.2 * state_size)
    alpha = 2
    Hh = []
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(state_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
        row = sorted(row)
        row = [y for (x, y) in row]
        Hh.append(row)
    Hh = np.asarray(Hh)

    # beta = np.random.uniform(0,1)
    beta = 1
    P_eq = np.multiply(beta, np.matmul(W,Hx)) + np.multiply((1-beta),Hh)

    ##finding the stationary distribution of h
    _, v = scipy.linalg.eig(P_eq, right=False, left=True)
    v = v.real
    st = v[:, 0] / sum(v[:, 0])
    print "sum of stationary distribution is: ", sum(st)

    ##finding the entropy of the source
    entropy = 0
    for i in range(state_size):
        ent = 0
        for j in range(vocab_size):
            if W[i, j] > 0:
                ent = ent + W[i, j] * np.log(W[i, j])
        ent = ent * st[i]
        entropy = entropy + ent
    entropy = -1 * entropy

    emp_entropy = 0

    # h_entropy = 0
    # for i in range(len(st)):
    #     if st[i] > 0:
    #         h_entropy = h_entropy + st[i] * np.log(st[i])
    # h_entropy = -1 * h_entropy

    print "entropy of source is: ", entropy

    h_train_seq = [0] * train_seq_len
    x_train_seq = [0] * train_seq_len
    h_test_seq = [0] * test_seq_len
    x_test_seq = [0] * test_seq_len
    h_valid_seq = [0] * valid_seq_len
    x_valid_seq = [0] * valid_seq_len

    x_train_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[0])
    x_test_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[1])
    x_valid_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[2])

    h_train_seq[0] = np.random.choice(np.arange(0, state_size),p=Hx[x_train_seq[0]])
    h_valid_seq[0] = np.random.choice(np.arange(0, state_size), p=Hx[x_valid_seq[0]])
    h_test_seq[0] = np.random.choice(np.arange(0, state_size), p=Hx[x_test_seq[0]])

    for i in range(1,train_seq_len):
        idx = h_train_seq[i-1]
        samp_p = W[idx]
        for j in range(vocab_size):
            if samp_p[j] > 0:
                emp_entropy = emp_entropy + samp_p[j] * np.log(samp_p[j])
        x_train_seq[i] = np.random.choice(np.arange(0, vocab_size), p=samp_p)
        idx = x_train_seq[i]
        hidx = h_train_seq[i-1]
        h_train_seq[i] = np.random.choice(np.arange(0, state_size), p=(beta*Hx[idx]+(1-beta)*Hh[hidx]))
    emp_entropy = -1.0 * emp_entropy
    emp_entropy = emp_entropy /train_seq_len
    print ("empirical entropy is: ", emp_entropy)
    for i in range(1,valid_seq_len):
        idx = h_valid_seq[i-1]
        x_valid_seq[i] = np.random.choice(np.arange(0, vocab_size), p=W[idx])
        idx = x_valid_seq[i]
        hidx = h_valid_seq[i-1]
        h_valid_seq[i] = np.random.choice(np.arange(0, state_size), p=(beta*Hx[idx]+(1-beta)*Hh[hidx]))

    for i in range(1,test_seq_len):
        idx = h_test_seq[i-1]
        x_test_seq[i] = np.random.choice(np.arange(0, vocab_size), p=W[idx])
        idx = x_test_seq[i]
        hidx = h_test_seq[i-1]
        h_test_seq[i] = np.random.choice(np.arange(0, state_size), p=(beta*Hx[idx] +(1-beta)*Hh[hidx]))

    print ("h_train: ", h_train_seq[90:110])
    print ("x_train: ", x_train_seq[90:110])
    return W, Hx, Hh, beta, st, entropy, x_train_seq, x_valid_seq, x_test_seq


def generate_syn_nonlinear(vocab_size, state_size, train_seq_len,
                       test_seq_len, valid_seq_len,percentile):
    shift = int(0.2 * vocab_size)
    alpha = 1.3
    W = []
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(vocab_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
        row = sorted(row)
        row = [y for (x, y) in row]
        W.append(row)
    W = np.asarray(W)

    shift = int(0.15 * vocab_size)
    alpha = 1.8
    Hx = []
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(state_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(vocab_size):
        row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
        row = sorted(row)
        row = [y for (x, y) in row]
        Hx.append(row)
    Hx = np.asarray(Hx)

    shift = int(0.25 * vocab_size)
    alpha = 2
    Hh = []
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(state_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x + np.random.randint(-shift, shift), y) for (x, y) in cand]
        row = sorted(row)
        row = [y for (x, y) in row]
        Hh.append(row)
    Hh = np.asarray(Hh)


    # Hx = np.random.uniform(0,10,size = [vocab_size,state_size])
    # Hh =np.random.uniform(0,10,size=[state_size,state_size])
    # Hx = Hx /Hx.sum(axis=1,keepdims=True)
    # Hh = Hh / Hh.sum(axis=1,keepdims=True)

    beta = np.random.uniform(0,1)

    P_eq = np.multiply(beta, np.matmul(W,Hx)) + np.multiply((1-beta),Hh)

    ##finding the stationary distribution of h
    _, v = scipy.linalg.eig(P_eq, right=False, left=True)
    v = v.real
    st = v[:, 0] / sum(v[:, 0])
    print "sum of stationary distribution is: ", sum(st)

    ##finding the entropy of the source
    # entropy = 0
    # for i in range(state_size):
    #     ent = 0
    #     for j in range(vocab_size):
    #         if W[i, j] > 0:
    #             ent = ent + W[i, j] * np.log(W[i, j])
    #     ent = ent * st[i]
    #     entropy = entropy + ent
    # entropy = -1 * entropy
    #
    # print "entropy of source is: ", entropy

    h_train_vec = [1.0] * state_size
    h_train_vec = np.array(h_train_vec) / state_size
    x_train_seq = [0] * train_seq_len
    h_test_vec = [1.0] * state_size
    h_test_vec = np.array(h_test_vec) / state_size
    x_test_seq = [0] * test_seq_len
    h_valid_vec = [1.0] * state_size
    h_valid_vec = np.array(h_valid_vec) / state_size
    x_valid_seq = [0] * valid_seq_len


    for i in range(1,train_seq_len):
        samp_p = np.matmul(h_train_vec,W)
        x_train_seq[i] = np.random.choice(np.arange(0, vocab_size), p=samp_p)
        idx = x_train_seq[i]
        h_train_vec = np.multiply(beta,Hx[idx]) + np.multiply((1-beta),np.matmul(h_train_vec,Hh))
        h_train_vec = non_linear_map(h_train_vec,percentile)

    for i in range(1,valid_seq_len):
        samp_p = np.matmul(h_valid_vec,W)
        x_valid_seq[i] = np.random.choice(np.arange(0, vocab_size), p=samp_p)
        idx = x_valid_seq[i]
        h_valid_vec = np.multiply(beta, Hx[idx]) + np.multiply((1 - beta), np.matmul(h_valid_vec, Hh))
        h_valid_vec = non_linear_map(h_valid_vec, percentile)

    count  = 0
    test_emp_entropy = 0
    for i in range(1,test_seq_len):
        samp_p = np.matmul(h_test_vec,W)
        if i > test_seq_len/2:
            count  = count + 1
            for j in range(vocab_size):
                if samp_p[j] > 0:
                    test_emp_entropy = test_emp_entropy + samp_p[j] * np.log(samp_p[j])
        x_test_seq[i] = np.random.choice(np.arange(0, vocab_size), p=samp_p)
        idx = x_test_seq[i]
        h_test_vec = np.multiply(beta, Hx[idx]) + np.multiply((1 - beta), np.matmul(h_test_vec, Hh))
        h_test_vec = non_linear_map(h_test_vec, percentile)
    test_emp_entropy = -1.0 * test_emp_entropy / count
    print ("empirical entropy of test is: ", test_emp_entropy)
    return W, Hx, Hh, beta, st, test_emp_entropy, x_train_seq, x_valid_seq, x_test_seq


##genrating data based on impaired recurrent model, or hmm

def generate_syn_hmm(vocab_size, state_size, train_seq_len,
                       test_seq_len, valid_seq_len):
    alpha = 1.3
    W = []
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(vocab_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x + np.random.randint(-200, 200), y) for (x, y) in cand]
        row = sorted(row)
        row = [y for (x, y) in row]
        W.append(row)
    W = np.asarray(W)


    # Hx = np.random.uniform(0,10,size = [vocab_size,state_size])
    Hh =np.random.uniform(0,10,size=[state_size,state_size])

    # Hx = Hx /Hx.sum(axis=1,keepdims=True)
    Hh = Hh / Hh.sum(axis=1,keepdims=True)
    # beta = np.random.uniform(0,1)

    # P_eq = np.multiply(beta, np.matmul(W,Hx)) + np.multiply((1-beta),Hh)

    #finding the stationary distribution of h
    _, v = scipy.linalg.eig(Hh, right=False, left=True)
    v = v.real
    st = v[:, 0] / sum(v[:, 0])
    print "sum of stationary distribution is: ", sum(st)

    #finding the entropy of the source
    entropy = 0
    for i in range(state_size):
        ent = 0
        for j in range(vocab_size):
            if W[i, j] > 0:
                ent = ent + W[i, j] * np.log(W[i, j])
        ent = ent * st[i]
        entropy = entropy + ent
    entropy = -1 * entropy

    h_entropy = 0
    for i in range(len(st)):
        if st[i] > 0:
            h_entropy = h_entropy + st[i] * np.log(st[i])
    h_entropy = -1 * h_entropy

    print "entropy of source is: ", entropy



    h_train_seq = [0] * train_seq_len
    x_train_seq = [0] * train_seq_len
    h_test_seq = [0] * test_seq_len
    x_test_seq = [0] * test_seq_len
    h_valid_seq = [0] * valid_seq_len
    x_valid_seq = [0] * valid_seq_len


    h_train_seq[0] = 0
    h_valid_seq[0] = 1
    h_test_seq[0] = 2

    x_train_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[0])
    x_test_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[1])
    x_valid_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[2])


    # h_train_seq[0] = np.random.choice(np.arange(0, state_size),p=Hx[x_train_seq[0]])
    # h_valid_seq[0] = np.random.choice(np.arange(0, state_size), p=Hx[x_valid_seq[0]])
    # h_test_seq[0] = np.random.choice(np.arange(0, state_size), p=Hx[x_test_seq[0]])

    for i in range(1,train_seq_len):
        idx = h_train_seq[i-1]
        x_train_seq[i] = np.random.choice(np.arange(0, vocab_size), p=W[idx])
        # idx = x_train_seq[i]
        hidx = h_train_seq[i-1]
        h_train_seq[i] = np.random.choice(np.arange(0, state_size), p=Hh[hidx])

    for i in range(1,valid_seq_len):
        idx = h_valid_seq[i-1]
        x_valid_seq[i] = np.random.choice(np.arange(0, vocab_size), p=W[idx])
        # idx = x_valid_seq[i]
        hidx = h_valid_seq[i-1]
        h_valid_seq[i] = np.random.choice(np.arange(0, state_size), p=Hh[hidx])

    for i in range(1,test_seq_len):
        idx = h_test_seq[i-1]
        x_test_seq[i] = np.random.choice(np.arange(0, vocab_size), p=W[idx])
        # idx = x_test_seq[i]
        hidx = h_test_seq[i-1]
        h_test_seq[i] = np.random.choice(np.arange(0, state_size), p=Hh[hidx])


    return W,Hh,st, entropy, x_train_seq, x_valid_seq, x_test_seq

##generates a sequence based on the hidden-markov model, h_t  -> h_t+1, and h_t -> x_t
def generate_syn_hmm_nonlinear(vocab_size, state_size, train_seq_len,
                       test_seq_len, valid_seq_len,percentile):
    alpha = 1.3
    W =[]
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(vocab_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x+np.random.randint(-100,100),y) for (x,y) in cand]
        row = sorted(row)
        row =[y for (x,y) in row]
        W.append(row)
    W = np.asarray(W)
    H = np.random.uniform(0,10, size = [state_size,state_size])
    H = H /H.sum(axis=1,keepdims=True)
    # P = np.matmul(H, W)

    ##finding the stationary distribution
    _, v = scipy.linalg.eig(H, right=False, left=True)
    v = v.real
    st = v[:, 0] / sum(v[:, 0])
    print "sum of stationary distribution is: ", sum(st)


    ##finding the entropy of the source
    entropy = 0
    # for i in range(state_size):
    #     ent = 0
    #     for j in range(vocab_size):
    #         if W[i, j] > 0:
    #             ent = ent + W[i, j] * np.log(W[i, j])
    #     ent = ent * st[i]
    #     entropy = entropy + ent
    # entropy = -1 * entropy
    #
    # h_entropy = 0
    # for i in range(len(st)):
    #     if st[i] > 0:
    #         h_entropy = h_entropy + st[i] * np.log(st[i])
    # h_entropy = -1 * h_entropy



    h_train_vec = [1.0] * state_size
    h_train_vec = np.array(h_train_vec) / state_size
    x_train_seq = [0] * train_seq_len
    h_test_vec = [1.0] * state_size
    h_test_vec = np.array(h_test_vec) /state_size
    x_test_seq = [0] * test_seq_len
    h_valid_vec = [1.0] * state_size
    h_valid_vec = np.array(h_valid_vec) / state_size
    x_valid_seq = [0] * valid_seq_len




    for i in range(0,train_seq_len):
        samp_p = np.matmul(h_train_vec,W)
        for j in range(vocab_size):
            if samp_p[j] > 0:
                entropy = entropy + samp_p[j] * np.log(samp_p[j])
        x_train_seq[i] = np.random.choice(np.arange(0, vocab_size), p=samp_p)

        h_train_vec = np.matmul(h_train_vec,H)
        h_train_vec = non_linear_map(h_train_vec,percentile)

    entropy = -1 * entropy
    entropy = entropy / train_seq_len
    print "entropy of source is: ", entropy

    for i in range(0,valid_seq_len):

        samp_p = np.matmul(h_valid_vec, W)
        # samp_p = non_linear_map(samp_p, percentile)
        x_valid_seq[i] = np.random.choice(np.arange(0, vocab_size), p=samp_p)

        h_valid_vec = np.matmul(h_valid_vec,H)
        h_valid_vec = non_linear_map(h_valid_vec, percentile)

    for i in range(0,test_seq_len):
        samp_p = np.matmul(h_test_vec, W)
        # samp_p = non_linear_map(samp_p, percentile)
        x_test_seq[i] = np.random.choice(np.arange(0, vocab_size), p=samp_p)

        h_test_vec = np.matmul(h_test_vec,H)
        h_test_vec = non_linear_map(h_test_vec, percentile)

    return W,H,st, entropy, x_train_seq, x_valid_seq, x_test_seq


##generate a sequence, and then returns the full sequence
def generate_syn_lr(vocab_size, state_size, train_seq_len,
                       test_seq_len, valid_seq_len):
    alpha = 1.3
    W =[]
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(vocab_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x+np.random.randint(-200,200),y) for (x,y) in cand]
        row = sorted(row)
        row =[y for (x,y) in row]
        W.append(row)
    W = np.asarray(W)
    H = np.random.uniform(0,10, size = [vocab_size,state_size])
    H = H /H.sum(axis=1,keepdims=True)
    P = np.matmul(H, W)

    ##finding the stationary distribution
    _, v = scipy.linalg.eig(P, right=False, left=True)
    v = v.real
    st = v[:, 0] / sum(v[:, 0])
    print "sum of stationary distribution is: ", sum(st)


    ##finding the entropy of the source
    entropy = 0
    for i in range(vocab_size):
        ent = 0
        for j in range(vocab_size):
            if P[i, j] > 0:
                ent = ent + P[i, j] * np.log(P[i, j])
        ent = ent * st[i]
        entropy = entropy + ent
    entropy = -1 * entropy
    print "entropy of source is: ", entropy

    h_train_seq = [0] * train_seq_len
    x_train_seq = [0] * train_seq_len
    h_test_seq = [0] * test_seq_len
    x_test_seq = [0] * test_seq_len
    h_valid_seq = [0] * valid_seq_len
    x_valid_seq = [0] * valid_seq_len

    x_train_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[0])
    x_test_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[1])
    x_valid_seq[0] = np.random.choice(np.arange(0, vocab_size), p=W[2])


    h_train_seq[0] = np.random.choice(np.arange(0, state_size),p=H[x_train_seq[0]])
    h_valid_seq[0] = np.random.choice(np.arange(0, state_size), p=H[x_valid_seq[0]])
    h_test_seq[0] = np.random.choice(np.arange(0, state_size), p=H[x_test_seq[0]])

    for i in range(1,train_seq_len):
        idx = h_train_seq[i-1]
        x_train_seq[i] = np.random.choice(np.arange(0, vocab_size), p=W[idx])
        idx = x_train_seq[i]
        h_train_seq[i] = np.random.choice(np.arange(0, state_size), p=H[idx])

    for i in range(1,valid_seq_len):
        idx = h_valid_seq[i-1]
        x_valid_seq[i] = np.random.choice(np.arange(0, vocab_size), p=W[idx])
        idx = x_valid_seq[i]
        h_valid_seq[i] = np.random.choice(np.arange(0, state_size), p=H[idx])

    for i in range(1,test_seq_len):
        idx = h_test_seq[i-1]
        x_test_seq[i] = np.random.choice(np.arange(0, vocab_size), p=W[idx])
        idx = x_test_seq[i]
        h_test_seq[i] = np.random.choice(np.arange(0, state_size), p=H[idx])

    return W,H,st, entropy, x_train_seq, x_valid_seq, x_test_seq


##generate a sequence based on non-linear low-rank model, x_t->h_t->x_t+1->h_t_+1
def generate_syn_lr_nonlinear(vocab_size, state_size, train_seq_len,
                       test_seq_len, valid_seq_len,percentile):
    alpha = 1.3
    W =[]
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(vocab_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x+np.random.randint(-100,100),y) for (x,y) in cand]
        row = sorted(row)
        row =[y for (x,y) in row]
        W.append(row)
    W = np.asarray(W)
    H = np.random.uniform(0,10, size = [vocab_size,state_size])
    H = H /H.sum(axis=1,keepdims=True)
    P = np.matmul(H, W)

    ##finding the stationary distribution
    _, v = scipy.linalg.eig(P, right=False, left=True)
    v = v.real
    st = v[:, 0] / sum(v[:, 0])
    print "sum of stationary distribution is: ", sum(st)


    ##finding the entropy of the source
    entropy = 0
    for i in range(vocab_size):
        ent = 0
        for j in range(vocab_size):
            if P[i, j] > 0:
                ent = ent + P[i, j] * np.log(P[i, j])
        ent = ent * st[i]
        entropy = entropy + ent
    entropy = -1 * entropy
    print "entropy of source is: ", entropy

    h_train_vec = [1.0] * state_size
    h_train_vec = np.array(h_train_vec) / state_size
    x_train_seq = [0] * train_seq_len
    h_test_vec = [1.0] * state_size
    h_test_vec = np.array(h_test_vec) /state_size
    x_test_seq = [0] * test_seq_len
    h_valid_vec = [1.0] * state_size
    h_valid_vec = np.array(h_valid_vec) / state_size
    x_valid_seq = [0] * valid_seq_len




    for i in range(0,train_seq_len):
        samp_p = np.matmul(h_train_vec,W)
        samp_p = non_linear_map(samp_p,60)
        x_train_seq[i] = np.random.choice(np.arange(0, vocab_size), p=samp_p)

        h_train_vec = np.matmul(samp_p,H)
        h_train_vec = non_linear_map(h_train_vec,60)

    for i in range(0,valid_seq_len):

        samp_p = np.matmul(h_valid_vec, W)
        samp_p = non_linear_map(samp_p, 60)
        x_valid_seq[i] = np.random.choice(np.arange(0, vocab_size), p=samp_p)

        h_valid_vec = np.matmul(samp_p,H)
        h_valid_vec = non_linear_map(h_valid_vec, 60)

    for i in range(0,test_seq_len):
        samp_p = np.matmul(h_test_vec, W)
        samp_p = non_linear_map(samp_p, 60)
        x_test_seq[i] = np.random.choice(np.arange(0, vocab_size), p=samp_p)

        h_test_vec = np.matmul(samp_p,H)
        h_test_vec = non_linear_map(h_test_vec, 60)

    return P,st, entropy, x_train_seq, x_valid_seq, x_test_seq



################## Extraaaaaas ############
###########################################

##generates independent pairs (bi-grams) and returns bi-grams
def generate_syn_non_recur_mixed(vocab_size, state_size, train_seq_len,
                       test_seq_len, valid_seq_len):
    alpha = 1.05
    W =[]
    for _ in range(state_size):
        cand = [1.0 / ((i+1)**alpha) for i in np.random.permutation(vocab_size)]
        s = sum(cand)
        cand = [x/s for x in cand]
        W.append(cand)
    W=np.asarray(W)

    H = np.random.uniform(0,10, size = [vocab_size,state_size])


    H = H /H.sum(axis=1,keepdims=True)

    P=np.matmul(H,W)
    _,v = scipy.linalg.eig(P,right=False, left=True)
    v = v.real
    st = v[:,0]/sum(v[:,0])
    print "sum of stationary distribution is: ", sum(st)

    entropy = 0
    for i in range(vocab_size):
        ent = 0
        for j in range(vocab_size):
            if P[i,j] >0:
                ent = ent + P[i,j]*np.log(P[i,j])
        ent = ent * st[i]
        entropy = entropy + ent
    entropy = -1* entropy
    print "entropy of source is: ", entropy

    x_train_seq=[]
    x_valid_seq=[]
    x_test_seq=[]

    for i in range(train_seq_len):
        x1 = np.random.choice(np.arange(0,vocab_size),p=st)
        x2= np.random.choice(np.arange(0,vocab_size),p=P[x1])
        x_train_seq.append((x1,x2))

    for i in range(valid_seq_len):
        x1 = np.random.choice(np.arange(0, vocab_size), p=st)
        x2 = np.random.choice(np.arange(0, vocab_size), p=P[x1])
        x_valid_seq.append((x1, x2))

    for i in range(test_seq_len):
        x1 = np.random.choice(np.arange(0, vocab_size), p=st)
        x2 = np.random.choice(np.arange(0, vocab_size), p=P[x1])
        x_test_seq.append((x1, x2))

    return W,H,entropy,st,x_train_seq, x_valid_seq, x_test_seq


##generate a sequence based on log-sum-exp of low-ranks
def generate_syn_lr_nonlinear_mixture(vocab_size, state_size, train_seq_len,
                       test_seq_len, valid_seq_len):
    alpha = 1.3
    W1 =[]
    cand = [(i, 1.0 / ((i + 1) ** alpha)) for i in range(vocab_size)]
    s = sum(y for (x, y) in cand)
    cand = [(x, y / s) for (x, y) in cand]
    for _ in range(state_size):
        row = [(x+np.random.randint(-50,50),y) for (x,y) in cand]
        row = sorted(row)
        row =[y for (x,y) in row]
        W1.append(row)
    W1 = np.asarray(W1)
    W2=W1
    W3=W1

    H1 = np.random.uniform(0,10, size = [vocab_size,state_size])
    H1 = H1 /H1.sum(axis=1,keepdims=True)

    H2 = np.random.uniform(0, 10, size=[vocab_size, state_size])
    H2 = H2 / H2.sum(axis=1, keepdims=True)

    H3 = np.random.uniform(0, 10, size=[vocab_size, state_size])
    H3 = H3 / H3.sum(axis=1, keepdims=True)

    H4 = np.random.uniform(0, 10, size=[vocab_size, state_size])
    H4 = H4 / H4.sum(axis=1, keepdims=True)

    P1 = np.matmul(H1,W1)
    P2 = np.matmul(H2,W2)
    P3 = np.matmul(H3,W1)
    P4 = np.matmul(H4,W1)


    beta = np.random.rand(4)
    beta = beta/sum(beta)

    P = np.log(beta[0]*np.exp(P1)+beta[1]*np.exp(P2)+beta[2]*np.exp(P3)+beta[3]*np.exp(P4))
    P = P/P.sum(axis=1, keepdims=True)


    ##finding the stationary distribution
    _, v = scipy.linalg.eig(P, right=False, left=True)
    v = v.real
    st = v[:, 0] / sum(v[:, 0])
    print "sum of stationary distribution is: ", sum(st)


    ##finding the entropy of the source
    entropy = 0
    for i in range(vocab_size):
        ent = 0
        for j in range(vocab_size):
            if P[i, j] > 0:
                ent = ent + P[i, j] * np.log(P[i, j])
        ent = ent * st[i]
        entropy = entropy + ent
    entropy = -1 * entropy
    print "entropy of source is: ", entropy

    # h_train_seq = [0] * train_seq_len
    x_train_seq = [0] * train_seq_len
    # h_test_seq = [0] * test_seq_len
    x_test_seq = [0] * test_seq_len
    # h_valid_seq = [0] * valid_seq_len
    x_valid_seq = [0] * valid_seq_len

    x_train_seq[0] = np.random.choice(np.arange(0, vocab_size))
    x_test_seq[0] = np.random.choice(np.arange(0, vocab_size))
    x_valid_seq[0] = np.random.choice(np.arange(0, vocab_size))


    # h_train_seq[0] = np.random.choice(np.arange(0, state_size),p=H1[x_train_seq[0]])
    # h_valid_seq[0] = np.random.choice(np.arange(0, state_size), p=H1[x_valid_seq[0]])
    # h_test_seq[0] = np.random.choice(np.arange(0, state_size), p=H1[x_test_seq[0]])

    for i in range(1,train_seq_len):
        idx = x_train_seq[i-1]
        x_train_seq[i] = np.random.choice(np.arange(0, vocab_size), p=P[idx])
        # idx = x_train_seq[i]
        # h_train_seq[i] = np.random.choice(np.arange(0, state_size), p=H1[idx])

    for i in range(1,valid_seq_len):
        idx = x_valid_seq[i-1]
        x_valid_seq[i] = np.random.choice(np.arange(0, vocab_size), p=P[idx])
        # idx = x_valid_seq[i]
        # h_valid_seq[i] = np.random.choice(np.arange(0, state_size), p=H1[idx])

    for i in range(1,test_seq_len):
        idx = x_test_seq[i-1]
        x_test_seq[i] = np.random.choice(np.arange(0, vocab_size), p=P[idx])
        # idx = x_test_seq[i]
        # h_test_seq[i] = np.random.choice(np.arange(0, state_size), p=H1[idx])

    return P,st, entropy, x_train_seq, x_valid_seq, x_test_seq


def preProcess_old(filename,vocab_size,batch_size,total_series_length):
    unknown_token = 'UNKNOWN'
    f_read = open(filename,'r')
    word_list =[]
    for line in f_read:
        if len(word_list) >= total_series_length:
            break
        line = line.split()
        for word in line:
            word_list.append(word)
            if len(word_list) >= total_series_length:
                break
    print ("total number of words:", len(word_list))

    fdist = nltk.FreqDist(word_list)
    print ("unique words:", len(fdist.keys()))
    vocab = fdist.most_common(vocab_size-1)
    print ("the least common word:", vocab[-1])
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    text_seq = [w if w in word_to_index else unknown_token for w in word_list]
    x = [word_to_index[w] for w in text_seq]
    x= x[:total_series_length]
    x = np.array(x)
    y = np.roll(x,1)
    y[0] = 0
    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))
    return (x,y)





