from RnnModel import GatedKneserNey
from utils import gatedPreProcess
import tensorflow as tf
import numpy as np
import time


def print_time():
    print '--- time: ', (time.time() - start_time) // 60,'minutes',\
        (time.time() - start_time) % 60, 'seconds'


def train_with_minibatch(train_data, test_data, model_filename, passes, batch_size, test_batch_size):

    model_path = model_filename
    with tf.Session() as sess:
        my_model = GatedKneserNey(session=sess,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  embedding_dim=embedding_dim,
                                  hidden_dim1=hidden_dim1,
                                  hidden_dim2=hidden_dim2,
                                  hidden_dim1_p=hidden_dim1_p,
                                  hidden_dim2_p=hidden_dim2_p,
                                  num_classes=classes_size,
                                  regularizer=reg)

        prev_valid_perplexity = None
        decay = 0
        valid_perplexities = list()

        for pass_number in range(passes):
            print 'pass#: ', pass_number + 1,  print_time()
            total_train_loss = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                sample_ys = [train_data[i][1] for i in range(start_idx, end_idx)]
                inputs = [train_data[i][0] for i in range(start_idx, end_idx)]
                batch_cost = my_model.batch_optimization(
                    inputs=inputs, labels=sample_ys, keep_prob=keep_prob)
                total_train_loss = total_train_loss + batch_cost
                if batch_idx % print_frequency == 0:
                    print 'training loss of batch ', batch_idx, ' = ', batch_cost

                # printing cross-entropy on validation data frequently
                if batch_idx % valid_frequency == 0:
                    cumul_loss = 0
                    for i in range(test_num_batches):
                        start_idx = i * test_batch_size
                        end_idx = start_idx + test_batch_size
                        inputs = [test_data[x][0] for x in range(start_idx, end_idx)]
                        sample_ys = [test_data[x][1] for x in range(start_idx, end_idx)]
                        batch_cost = my_model.predict(
                            inputs=inputs,
                            labels=sample_ys,
                            keep_prob=1)
                        cumul_loss = cumul_loss + batch_cost
                    temp_valid_preplixity = cumul_loss / test_num_batches
                    print('validation data loss = ', temp_valid_preplixity)

            # printing cross-entropy on validation data at the end of every epoch
            total_valid_loss = 0
            for i in range(test_num_batches):
                start_idx = i * test_batch_size
                end_idx = start_idx + test_batch_size
                sample_ys = [test_data[x][1] for x in range(start_idx, end_idx)]
                inputs = [test_data[x][0] for x in range(start_idx, end_idx)]
                batch_cost = my_model.predict(
                    inputs=inputs,
                    labels=sample_ys,
                    keep_prob=1)
                total_valid_loss = total_valid_loss + batch_cost

            valid_perplexity = total_valid_loss / test_num_batches
            print('validation data loss = %f', valid_perplexity)
            if prev_valid_perplexity != None and valid_perplexity - best_valid_perplexity > 0:
                my_model.learning_rate = my_model.learning_rate / 2
                decay = decay + 1
                print("loading epoch %d parameters, perplexity %f", best_epoch, best_valid_perplexity)
                my_model._saver.restore(sess, model_path + '_epoch%d.ckpt' % best_epoch)

            prev_valid_perplexity = valid_perplexity
            valid_perplexities.append(valid_perplexity)
            if valid_perplexity <= np.min(valid_perplexities):
                best_epoch = pass_number
                best_valid_perplexity = valid_perplexity
                save_path = my_model._saver.save(sess, model_path + '_epoch%d.ckpt' % pass_number)
                print('saved model to file: %s', save_path)

            if decay > max_learning_rate_decay:
                print("reached maximum decay number, quitting after epoch%d", pass_number)
                break

    return best_valid_perplexity, best_epoch

def predict(test_data, model_path, best_epoch):

    tf.reset_default_graph()

    with tf.Session() as sess:

        my_model = GatedKneserNey(session=sess,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  embedding_dim=embedding_dim,
                                  hidden_dim1=hidden_dim1,
                                  hidden_dim2=hidden_dim2,
                                  hidden_dim1_p=hidden_dim1_p,
                                  hidden_dim2_p=hidden_dim2_p,
                                  num_classes=classes_size,
                                  regularizer=reg)

        my_model._saver.restore(sess,  model_path + '_epoch%d.ckpt' % best_epoch)
        cumul_loss = 0
        for batch_idx in range(test_num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            sample_ys = [test_data[x][1] for x in range(start_idx,end_idx)]
            inputs = [test_data[x][0] for x in range(start_idx,end_idx)]
            batch_cost = my_model.predict(
                inputs=inputs,
                labels=sample_ys,keep_prob=1)
            cumul_loss = cumul_loss + batch_cost
        print  'test cross entropy:', cumul_loss/test_num_batches

        #########################################################

# Main body of the script

start_time = time.time()

# data file names, output paths ##############
train_filename = '../data/ptb/ptb.train.txt'
validation_filename = '../data/ptb/ptb.valid.txt'
test_filename = '../data/ptb/ptb.test.txt'

# training parameters ##################
passes = 30
batch_size = 256  # for training
test_batch_size = batch_size
learning_rate = 0.005
embedding_dim = 64
hidden_dim1 = 128
hidden_dim2 = 128
hidden_dim1_p = 32
hidden_dim2_p = 32
max_learning_rate_decay = 8
keep_prob = 0.7
reg = 0.0
classes_size = 10000
print_frequency = 100
valid_frequency = 1000

# path for final trained model
model_filename = './saved_models/gatedKneserNey/' + str(hidden_dim1) +\
                 '_' + str(hidden_dim2)+'_' + str(keep_prob) + '_' + str(reg)

print("extracting train and test inputs and labels as a very long sequence of words")
train_data, test_data, _ = gatedPreProcess(train_filename, validation_filename,
                                                    test_filename, classes_size, batch_size)
num_batches = len(train_data) // batch_size
test_num_batches = len(test_data) // batch_size
# validation_num_batches = validation_length //batch_size
print 'Training using Neural Networks'
_, best_epoch = train_with_minibatch(train_data,
                                    test_data,
                                    model_filename,
                                    passes,
                                    batch_size,
                                    test_batch_size)
print '---- time: ', print_time()
print 'Testing on test data...'
predict(test_data, model_filename, best_epoch)
print '---- time: ', print_time()

