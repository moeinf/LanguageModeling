from __future__ import print_function, division
from utils import preProcess
import numpy as np
import tensorflow as tf
from RnnModel import TfMultiCellLSTM
import time
start_time = time.time()

###Parameters###
num_epochs = 5
total_series_length = 887520
test_total_series_length = 78660
truncated_backprop_length = 35
learning_rate = 0.5
state_size = 50
num_layers = 2
vocab_size = 10000
test_vocab_size = 3039
print_freq = 100
echo_step = 1
batch_size = 20
test_batch_size = 20
model_filename = './saved_models/MultiLayer_LSTM_model'  #path for final trained model
train_filename = 'ptb.train.txt'
test_filename = 'ptb.test.txt'

num_batches = total_series_length//batch_size//truncated_backprop_length
test_num_batches = test_total_series_length //test_batch_size // truncated_backprop_length
model_path = model_filename

#function to print time
def print_time():
    print ('--- time: ', (time.time() - start_time) // 60, \
        'minutes', (time.time() - start_time) % 60, 'seconds')

#the function does the training on train data, using mini-batches and
#report the loss on the whole test data set frequently

def train_with_batches(train_data_filename,test_data_filename):

    #extracting train and test inputs and labels as a very long sequence of words
    x, y, x_test, y_test = preProcess(train_data_filename,test_data_filename, vocab_size, batch_size,
                                      test_batch_size, total_series_length)

    with tf.Session() as sess:
        loss_list = []
        myRnn = TfMultiCellLSTM(session=sess,
                        learning_rate=learning_rate,
                        state_size=state_size,
                        num_classes=vocab_size,
                        num_layers=num_layers,
                        truncated_backprop_length=truncated_backprop_length,
                        batch_size= batch_size,
                        series_length=total_series_length)

        for epoch_idx in range(num_epochs):
            print ('pass#: ', epoch_idx + 1, '---- time: ', print_time())


            state = np.zeros((num_layers, 2, batch_size, state_size))  #initial state of multi-layer LSTM

            for batch_idx in range(num_batches):
                #extracting each batch for training data
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x[:,start_idx:end_idx]
                batchY = y[:,start_idx:end_idx]

                batch_cost, state = myRnn.batch_optimiztion(batchX,batchY,state)
                loss_list.append(batch_cost)
                # predictions, single_predictions, batch_cost = myRnn.predict(batchX,batchY,states)



                if batch_idx% print_freq == 0:
                    print("Step",batch_idx, "Loss", batch_cost)

                    #calculate the loss function of the so-far trained model on the test data
                    cumul_loss = 0
                    current_test_state = np.zeros((num_layers, 2, batch_size, state_size))
                    #extracting batches from the test data (it can also be done not in batches)
                    for batch_idx in range(test_num_batches):
                        start_idx = batch_idx * truncated_backprop_length
                        end_idx = start_idx + truncated_backprop_length

                        batchX = x_test[:, start_idx:end_idx]
                        batchY = y_test[:, start_idx:end_idx]

                        predictions, single_predictions, test_batch_cost, current_test_state = myRnn.predict(batchX,
                                                                                                             batchY,
                                                                                                             current_test_state)
                        cumul_loss = cumul_loss + test_batch_cost

                    print('loss on testing:', cumul_loss / test_num_batches)

                # save the model for later use
                if model_path:
                    myRnn._saver.save(sess, model_path)

#this function is written to do the testing after the training is completely done
#further we can manually stop the training at any time and use the so-far trained model
#stored in model_path for evaluation on the test set
def test_with_batches(train_data_filename,test_data_filename):

    tf.reset_default_graph()
    _, _, x_test, y_test = preProcess(train_data_filename, test_data_filename, test_vocab_size, batch_size,
                                      test_batch_size, total_series_length)

    with tf.Session() as sess:
        myRnn = TfMultiCellLSTM(session=sess,
                        learning_rate=learning_rate,
                        state_size=state_size,
                        num_classes=test_vocab_size,
                        num_layers=num_layers,
                        truncated_backprop_length=truncated_backprop_length,
                        batch_size=test_batch_size,
                        series_length=test_total_series_length)
        myRnn._saver.restore(sess, model_path)          #restoring the model from the last checkpoint

        current_test_state = np.zeros((num_layers, 2, batch_size, state_size))  # initial state of multi-layer LSTM
        #calculate the loss on the test data in mini-batches, it can also be done not in batches
        cumul_loss = 0

        #extracting batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x_test[:, start_idx:end_idx]
            batchY = y_test[:, start_idx:end_idx]

            predictions, single_predictions, batch_cost, current_test_state = myRnn.predict(batchX, batchY, current_test_state)
            cumul_loss = cumul_loss + batch_cost

        print ("testing loss:", cumul_loss/num_batches)


train_with_batches(train_filename, test_filename)
# print ("Testing begins:")
# test_with_batches(test_filename)