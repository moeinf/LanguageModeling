from __future__ import print_function, division
from utils import preProcess
import numpy as np
import tensorflow as tf
from RnnModel import TfMultiCellLSTM
import time
start_time = time.time()

###Parameters###
num_epochs = 3
truncated_backprop_length = 35
learning_rate = 0.5
state_size = 20
num_layers = 2
vocab_size = 10000
print_freq = 100
echo_step = 1
batch_size = 20
model_filename = './saved_models/MultiLayer_LSTM_model'+ '_' + str(truncated_backprop_length)+ '_' + str(state_size) + '_' + str(num_layers) #path for final trained model
train_filename = 'ptb.train.txt'
validation_filename = 'ptb.valid.txt'
test_filename = 'ptb.test.txt'

model_path = model_filename

#function to print time
def print_time():
    print ('--- time: ', (time.time() - start_time) // 60, \
        'minutes', (time.time() - start_time) % 60, 'seconds')

#the function does the training on train data, using mini-batches and
#report the loss on the whole test data set frequently
def train_with_batches(train_inputs,train_labels, validation_inputs, validation_labels):

    x = train_inputs
    y = train_labels
    x_validation = validation_inputs
    y_validation = validation_labels

    with tf.Session() as sess:
        myRnn = TfMultiCellLSTM(session=sess,
                        learning_rate=learning_rate,
                        state_size=state_size,
                        num_classes=vocab_size,
                        num_layers=num_layers,
                        truncated_backprop_length=truncated_backprop_length,
                        batch_size= batch_size)

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
                # predictions, single_predictions, batch_cost = myRnn.predict(batchX,batchY,states)



                if batch_idx% print_freq == 0:
                    print("Step",batch_idx, "Loss", batch_cost)
                    #calculate the loss function of the so-far trained model on the test data
                    cumul_loss = 0
                    current_validation_state = np.zeros((num_layers, 2, batch_size, state_size))
                    #extracting batches from the test data (it can also be done not in batches)
                    for batch_idx in range(validation_num_batches):
                        start_idx = batch_idx * truncated_backprop_length
                        end_idx = start_idx + truncated_backprop_length

                        batchX = x_validation[:, start_idx:end_idx]
                        batchY = y_validation[:, start_idx:end_idx]

                        _, _, validation_batch_cost, current_validation_state = myRnn.predict(batchX,
                                                                                              batchY,
                                                                                              current_validation_state)
                        cumul_loss = cumul_loss + validation_batch_cost

                    print('validation data loss:', cumul_loss / validation_num_batches)

                # save the model for later use
                if model_path:
                    myRnn._saver.save(sess, model_path)

#this function is written to do the testing after the training is completely done
#further we can manually stop the training at any time and use the so-far trained model
#stored in model_path for evaluation on the test set
def test_with_batches(test_inputs, test_labels):

    x_test = test_inputs
    y_test = test_labels

    tf.reset_default_graph()

    with tf.Session() as sess:
        myRnn = TfMultiCellLSTM(session=sess,
                        learning_rate=learning_rate,
                        state_size=state_size,
                        num_classes=vocab_size,
                        num_layers=num_layers,
                        truncated_backprop_length=truncated_backprop_length,
                        batch_size=batch_size)
        myRnn._saver.restore(sess, model_path)          #restoring the model from the last checkpoint

        current_test_state = np.zeros((num_layers, 2, batch_size, state_size))  # initial state of multi-layer LSTM
        #calculate the loss on the test data in mini-batches, it can also be done not in batches
        cumul_loss = 0

        #extracting batches
        for batch_idx in range(test_num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x_test[:, start_idx:end_idx]
            batchY = y_test[:, start_idx:end_idx]

            predictions, single_predictions, batch_cost, current_test_state = myRnn.predict(batchX, batchY, current_test_state)
            cumul_loss = cumul_loss + batch_cost

        print ("testing loss:", cumul_loss/test_num_batches)


print("extracting train and test inputs and labels as a very long sequence of words")
x, y, x_validation, y_validation, x_test, y_test, train_length, validation_length, test_length = preProcess(train_filename,
                                                             validation_filename,
                                                             test_filename,
                                                             vocab_size,
                                                             batch_size)

num_batches = train_length// batch_size//truncated_backprop_length
test_num_batches = test_length //batch_size // truncated_backprop_length
validation_num_batches = validation_length //batch_size // truncated_backprop_length
# print ("Training begins:")
train_with_batches(x,y,x_validation,y_validation)
print ("Testing begins:")
test_with_batches(x_test, y_test)