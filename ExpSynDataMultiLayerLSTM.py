from __future__ import print_function, division
from utils import preProcessSynthetic
import tensorflow as tf
from RnnModel import TfMultiCellLSTM
from RnnModel import TfMultiCellRNN
from datetime import datetime
import time
start_time = time.time()

###Parameters###
num_epochs = 100
truncated_backprop_length = 30
learning_rate = 0.005
state_size = 100
regularizer = 0.001
embedding_size = 100
num_layers = 2
vocab_size = 100
batch_size = 20
print_freq = 100
model_name = 'TfMultiCellRNN'

data_filename = '../data/syn/syn_data_unique.npz'
model_filename = './saved_models/Syn_unique'+ model_name+'_' +\
                 str(truncated_backprop_length)+ '_' + str(state_size) +\
                 '_' + str(num_layers) #path for final trained model


model_path = model_filename

#the function does the training on train data, using mini-batches and
#report the loss on the whole test data set frequently

def train_with_batches(train_inputs, train_labels, validation_inputs, validation_labels):

    cumul_loss = 0
    x = train_inputs
    y = train_labels
    x_validation = validation_inputs
    y_validation = validation_labels

    with tf.Session() as sess:
        myRnn = TfMultiCellRNN(session=sess,
                                learning_rate=learning_rate,
                                state_size=state_size,
                                num_classes=vocab_size,
                                num_layers=num_layers,
                                truncated_backprop_length=truncated_backprop_length,
                                embed_dim=embedding_size,
                                batch_size= batch_size,
                                regularizer=regularizer)

        for epoch_idx in range(num_epochs):

            print ('pass#: ', epoch_idx + 1)
            state = sess.run(myRnn._initial_state)tax

            for batch_idx in range(num_batches):

                #extracting each batch for training data
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x[:,start_idx:end_idx]
                batchY = y[:,start_idx:end_idx]

                batch_cost, state = myRnn.batch_optimiztion(batchX,
                                                            batchY,
                                                            state,
                                                            0.8)

                if batch_idx% print_freq == 0:
                    print(datetime.now(), "-->Training loss at step",batch_idx, "=",
                          batch_cost/truncated_backprop_length)
                    # calculate the loss function of the so-far trained model on
                    # the test data
                    prev_loss = cumul_loss
                    cumul_loss = 0
                    current_validation_state = sess.run(myRnn._initial_state)
                    #extracting batches from the test data (it can also be done not
                    # in mini-batches)
                    for i in range(validation_num_batches):
                        start_idx = i * truncated_backprop_length
                        end_idx = start_idx + truncated_backprop_length

                        batchX = x_validation[:, start_idx:end_idx]
                        batchY = y_validation[:, start_idx:end_idx]

                        validation_batch_cost, current_validation_state \
                            = myRnn.predict(batchX,
                                            batchY,
                                            current_validation_state,
                                            1.0)
                        cumul_loss = cumul_loss + validation_batch_cost

                    if cumul_loss - prev_loss > -0.01:
                        myRnn.learning_rate = myRnn.learning_rate/2
                        print ("learning rate is halved")
                    print(datetime.now(), '\033[1m'+'--> validation data loss:',
                          cumul_loss/validation_num_batches/truncated_backprop_length)
                    print ('\033[0m')
                # save the model for later use
                    if model_path:
                        myRnn._saver.save(sess, model_path)

#this function is written to do the testing after the training is completely done
#further we can manually stop the training at any time and use the so-far trained model
#stored in model_path for evaluation on the test set
def test_with_batches(test_inputs, test_labels):
    cumul_loss = 0
    x_test = test_inputs
    y_test = test_labels

    tf.reset_default_graph()

    with tf.Session() as sess:
        myRnn = TfMultiCellRNN(session=sess,
                        learning_rate=learning_rate,
                        state_size=state_size,
                        num_classes=vocab_size,
                        num_layers=num_layers,
                        truncated_backprop_length=truncated_backprop_length,
                        embed_dim=embedding_size,
                        batch_size=batch_size,
                        regularizer=regularizer)

        myRnn._saver.restore(sess, model_path) #restoring the model from
        # the last checkpoint

        current_test_state = sess.run(myRnn._initial_state) # initial state of
        # multi-layer LSTM

        #calculate the loss on the test data in mini-batches,
        # it can also be done not in batches
        prev_loss = cumul_loss
        cumul_loss = 0
        #extracting batches
        for batch_idx in range(test_num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x_test[:, start_idx:end_idx]
            batchY = y_test[:, start_idx:end_idx]

            batch_cost, current_test_state = myRnn.predict(batchX,
                                                           batchY,
                                                           current_test_state,
                                                           1)
            cumul_loss = cumul_loss + batch_cost
        if cumul_loss - prev_loss < -3:
            myRnn.learning_rate = myRnn.learning_rate / 2
            print ("learning rate is halved")
        print ("testing loss:", cumul_loss/test_num_batches/truncated_backprop_length)


print("preprocessing synthetic data")
x_train, y_train, x_valid, y_valid, x_tes, y_tes, train_length,\
valid_length,test_length = preProcessSynthetic(batch_size,data_filename)


print ("synthetic data preprocessing is done!")
num_batches = train_length// batch_size//truncated_backprop_length
test_num_batches = test_length // batch_size // truncated_backprop_length
validation_num_batches = valid_length //batch_size // truncated_backprop_length
print ("Training begins:")
train_with_batches(x_train,y_train,x_tes,y_tes)
print ("Testing begins:")
test_with_batches(x_tes, y_tes)