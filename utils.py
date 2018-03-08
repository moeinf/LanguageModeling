import numpy as np
import nltk
from random import shuffle

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


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

def preProcess(train_filename,validation_filename, test_filename, vocab_size,batch_size):
    unknown_token = 'UNKNOWN'
    f_read_train = open(train_filename,'r')
    f_read_validation = open(validation_filename,'r')
    f_read_test = open(test_filename,'r')
    train_word_list=[]
    validation_word_list = []
    test_word_list=[]
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

    x_validation = x_validation.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y_validation = y_validation.reshape((batch_size, -1))

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