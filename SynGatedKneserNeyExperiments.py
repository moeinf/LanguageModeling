from RnnModel import gatedKneserNey
from utils import gatedPreProcessSyn
import tensorflow as tf
import numpy as np
import time
start_time = time.time()
data_file='syn_data_nonlinear.npz'
model_filename = './saved_models/gatedKneserNeySynNonRecurNonLinear'  # path for final trained model

##### training parameters ##################
passes = 3
batch_size = 64  #for training
test_batch_size = 64


learning_rate = 0.001

hidden_dim1 = 256
hidden_dim2 = 128
classes_size = 1000
print_frequency = 128

def print_time():
    print '--- time: ', (time.time() - start_time) // 60, \
        'minutes', (time.time() - start_time) % 60, 'seconds'


def extract_words(data_batch,vocab_size):
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
                                      num_classes=classes_size)

        for pass_number in range(passes):
            print 'pass#: ', pass_number + 1 , '---- time: ',  print_time()

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                data_batch = train_data[start_idx:end_idx]
                sample_ys, inputs_indices, inputs_values, inputs_shape = \
                    extract_words(data_batch,classes_size)
                batch_cost,check,check_z,check_b,check_c = my_model.batch_optimization(
                    inputs=(inputs_indices,inputs_values,inputs_shape),labels=sample_ys,keep_prob=0.8)

                if batch_idx % print_frequency == 0:
                    print 'training loss of batch', batch_idx, ': = ', batch_cost
                    # print 'check_sum: ', check
                    # print 'check_sum for backoff is: ', check_b
                    # print 'check_sum for count is: ', check_c
                    # print 'check_sum for z is: ', check_z


                    cumul_loss = 0
                    for i in range(test_num_batches):
                        start_idx = i * test_batch_size
                        end_idx = start_idx + test_batch_size
                        data_batch = test_data[start_idx:end_idx]
                        sample_ys, inputs_indices, inputs_values, inputs_shape = \
                            extract_words(data_batch, classes_size)
                        batch_cost,_ = my_model.predict(
                            inputs=(inputs_indices, inputs_values, inputs_shape),
                            labels=sample_ys,
                            keep_prob=1)
                        cumul_loss = cumul_loss + batch_cost
                    print '\033[1m'+'testing loss after batch', batch_idx, ': = ', cumul_loss/test_num_batches
                    print '\033[0m'

                    if model_path:
                        my_model._saver.save(sess, model_path)



def predict(test_data, model_path):

    tf.reset_default_graph()

    with tf.Session() as sess:

        my_model = gatedKneserNey(session=sess,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  hidden_dim1=hidden_dim1,
                                  hidden_dim2=hidden_dim2,
                                  num_classes=classes_size)

        my_model._saver.restore(sess, model_path)
        cumul_loss = 0
        for batch_idx in range(test_num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            data_batch = test_data[start_idx:end_idx]
            sample_ys, inputs_indices, inputs_values, inputs_shape = extract_words(data_batch, classes_size)
            batch_cost,predictions = my_model.predict(
                inputs=(inputs_indices,inputs_values, inputs_shape), labels=sample_ys,keep_prob=1)
            cumul_loss = cumul_loss + batch_cost
        print  'test cross entropy:', cumul_loss/test_num_batches

        #########################################################

print("extracting train and test inputs and labels as a very long sequence of words")
###use this part for a sequence data
train_data, test_data, _, train_length, _, test_length = gatedPreProcessSyn(data_file,
                                                                            batch_size)

###use this part for independent pairs
# files = np.load(data_file)
# train_data = files['x_train']
# test_data = files['x_test']
#
# train_length = len(train_data) // batch_size * batch_size
# test_length = len(test_data) // batch_size * batch_size

num_batches = train_length // batch_size
test_num_batches = test_length // batch_size

print 'Training using Neural Networks'
train_with_minibatch(train_data,test_data,model_filename, passes, batch_size,test_batch_size)
print '---- time: ', print_time()
print 'Testing on test data...'
predict(test_data, model_filename)
print '---- time: ', print_time()
