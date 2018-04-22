from RnnModel import gatedKneserNey
from utils import gatedPreProcess
import tensorflow as tf
import numpy as np
import time

def print_time():
    print '--- time: ', (time.time() - start_time) // 60,'minutes',\
        (time.time() - start_time) % 60, 'seconds'
def extract_words(data_batch, vocab_size):
    batch_len = len(data_batch)
    shape = np.array([batch_len, vocab_size], dtype=np.int64)
    indices_array = []
    values_array = []
    # inputs = np.zeros([batch_len,vocab_size])
    label = np.zeros([batch_len])
    for batch_index in range(batch_len):
        input_idx = data_batch[batch_index][0]
        label_idx = data_batch[batch_index][1]
        label[batch_index] = label_idx
        indices_array.append([batch_index, input_idx])
        values_array.append(1.0)
        # inputs[batch_index][input_idx] = 1
    return label, np.array(indices_array, dtype=np.int64), np.array(values_array, dtype=np.float32), shape

def train_with_minibatch(train_data, test_data, model_filename, passes, batch_size,test_batch_size):
    model_path = model_filename
    with tf.Session() as sess:

        my_model = gatedKneserNey(session=sess,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  hidden_dim1=hidden_dim1,
                                  hidden_dim2=hidden_dim2,
                                  num_classes=classes_size,
                                  regularizer=reg)

        prev_valid_preplixity = None
        decay = 0
        valid_preplixities = list()

        for pass_number in range(passes):
            print 'pass#: ', pass_number + 1,  print_time()
            total_train_loss = 0
            for batch_idx in range(num_batches):

                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                data_batch = train_data[start_idx:end_idx]
                sample_ys, inputs_indices, inputs_values, inputs_shape = \
                    extract_words(data_batch,classes_size)
                batch_cost = my_model.batch_optimization(
                    inputs=(inputs_indices,inputs_values,inputs_shape),labels=sample_ys,keep_prob=keep_prob)

                total_train_loss = total_train_loss + batch_cost
                if batch_idx % print_frequency == 0:
                    print 'training loss of batch', batch_idx, ': = ', batch_cost
                if batch_idx % (5*print_frequency) == 0:
                    cumul_loss = 0
                    for i in range(test_num_batches):
                        start_idx = i * test_batch_size
                        end_idx = start_idx + test_batch_size
                        data_batch = test_data[start_idx:end_idx]
                        sample_ys, inputs_indices, inputs_values, inputs_shape = \
                            extract_words(data_batch, classes_size)
                        batch_cost= my_model.predict(
                            inputs=(inputs_indices, inputs_values, inputs_shape),
                            labels=sample_ys,
                            keep_prob=1)
                        cumul_loss = cumul_loss + batch_cost

                    temp_valid_preplixity = cumul_loss / test_num_batches
                    print('validation data loss = %f', temp_valid_preplixity)

            total_valid_loss = 0
            for i in range(test_num_batches):
                start_idx = i * test_batch_size
                end_idx = start_idx + test_batch_size
                data_batch = test_data[start_idx:end_idx]
                sample_ys, inputs_indices, inputs_values, inputs_shape = \
                    extract_words(data_batch, classes_size)
                batch_cost = my_model.predict(
                    inputs=(inputs_indices, inputs_values, inputs_shape),
                    labels=sample_ys,
                    keep_prob=1)
                total_valid_loss = total_valid_loss + batch_cost

            valid_preplixity = total_valid_loss / test_num_batches
            print('validation data loss = %f', valid_preplixity)
            if prev_valid_preplixity != None and valid_preplixity - best_valid_preplixity > 0:
                my_model.learning_rate = my_model.learning_rate / 2
                decay = decay + 1
                print("loading epoch %d parameters, preplixity %f", best_epoch, best_valid_preplixity)
                my_model._saver.restore(sess, model_path + '_epoch%d.ckpt' % best_epoch)

            prev_valid_preplixity = valid_preplixity
            valid_preplixities.append(valid_preplixity)
            if valid_preplixity <= np.min(valid_preplixities):
                best_epoch = pass_number
                best_valid_preplixity = valid_preplixity
                save_path = my_model._saver.save(sess, model_path + '_epoch%d.ckpt' % pass_number)
                print('saved model to file: %s', save_path)

            if decay > max_learning_rate_decay:
                print("reached maximum decay number, quitting after epoch%d", pass_number)
                break

    return best_valid_preplixity, best_epoch

def predict(test_data, model_path,best_epoch):

    tf.reset_default_graph()

    with tf.Session() as sess:

        my_model = gatedKneserNey(session=sess,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  hidden_dim1=hidden_dim1,
                                  hidden_dim2=hidden_dim2,
                                  num_classes=classes_size,
                                  regularizer=reg)

        my_model._saver.restore(sess,  model_path + '_epoch%d.ckpt' % best_epoch)
        cumul_loss = 0
        for batch_idx in range(test_num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            data_batch = test_data[start_idx:end_idx]
            sample_ys, inputs_indices, inputs_values, inputs_shape = extract_words(data_batch,
                                                                                   classes_size)
            batch_cost = my_model.predict(
                inputs=(inputs_indices,inputs_values, inputs_shape),
                labels=sample_ys,keep_prob=1)
            cumul_loss = cumul_loss + batch_cost
        print  'test cross entropy:', cumul_loss/test_num_batches

        #########################################################

## Main body of the script

start_time = time.time()


######## data file names, output paths ##############
train_filename = '../data/ptb/ptb.train.txt' # training data file
validation_filename = '../data/ptb/ptb.valid.txt'
test_filename = '../data/ptb/ptb.test.txt'

##### training parameters ##################
passes = 10
batch_size = 100  #for training
test_batch_size = 100

learning_rate = 0.01

hidden_dim1 = 1024
hidden_dim2 = 512

max_learning_rate_decay = 8

keep_prob = 0.7
reg = 0.0
classes_size = 10000
print_frequency = 100

# path for final trained model
model_filename = './saved_models/gatedKneserNey/'+ str(hidden_dim1) + '_' + str(hidden_dim2)+'_' + str(keep_prob)


print("extracting train and test inputs and labels as a very long sequence of words")
train_data, test_data, _, train_length, _,test_length = gatedPreProcess(train_filename,
                                                                        validation_filename,
                                                                        test_filename,
                                                                        classes_size,
                                                                        batch_size)

num_batches = train_length // batch_size
test_num_batches = test_length // batch_size
# validation_num_batches = validation_length //batch_size
print 'Training using Neural Networks'
_,best_epoch = train_with_minibatch(train_data,
                                    test_data,
                                    model_filename,
                                    passes,
                                    batch_size,
                                    test_batch_size)
print '---- time: ', print_time()
print 'Testing on test data...'
predict(test_data, model_filename,best_epoch)
print '---- time: ', print_time()

