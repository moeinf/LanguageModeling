from __future__ import print_function, division
from utils import preProcess
import tensorflow as tf
from RnnModel import SmoothedSoftmax
import logging
import numpy as np


###Parameters###
num_epochs = 30
truncated_backprop_length = 25
test_truncated_length = 1
learning_rate = 1
state_size = 256
embedding_size = 256
num_layers = 1
vocab_size = 10000
print_freq = 100
batch_size = 20
test_batch_size = 1
beta=0.5
max_learning_rate_decay = 8
smoothing = 'separate-softmax'
log_filename = './logs/'+smoothing+'.log'
train_filename = '../data/ptb/ptb.train.txt'
validation_filename = '../data/ptb/ptb.valid.txt'
test_filename = '../data/ptb/ptb.test.txt'
model_filename = './saved_models/'+smoothing+'/' + \
                 str(truncated_backprop_length) + '_' + str(
    state_size) + '_' + str(num_layers)
# path for final trained model

model_path = model_filename


# the function does the training on train data, using mini-batches and
# report the loss on the whole test data set frequently

def train_with_batches(train_inputs, train_labels, validation_inputs, validation_labels):
    x = train_inputs
    y = train_labels
    x_validation = validation_inputs
    y_validation = validation_labels

    with tf.Session() as sess:
        myRnn = SmoothedSoftmax(session=sess,
                                learning_rate=learning_rate,
                                state_size=state_size,
                                num_classes=vocab_size,
                                num_layers=num_layers,
                                truncated_backprop_length=truncated_backprop_length,
                                embed_dim=embedding_size,
                                batch_size=batch_size,
                                beta=beta,
                                smoothing=smoothing)

        prev_valid_preplixity = None
        decay = 0
        valid_preplixities = list()
        for epoch_idx in range(num_epochs):
            logging.info('pass %d: ', epoch_idx + 1)

            state = sess.run(myRnn._initial_state)

            for batch_idx in range(num_batches):
                # extracting each batch for training data
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x[:, start_idx:end_idx]
                batchY = y[:, start_idx:end_idx]

                batch_cost, state = myRnn.batch_optimiztion(batchX, batchY, state, 0.5)
                # predictions, single_predictions, batch_cost =
                # myRnn.predict(batchX,batchY,states)

                if batch_idx % print_freq == 0:
                    logging.info("Training loss at step %d = %f",
                                 batch_idx, batch_cost / truncated_backprop_length)

            # calculate the loss function of the so-far trained model on
            # the test data
            cumul_loss = 0
            current_validation_state = sess.run(myRnn._initial_state)
            # extracting batches from the test data (it can also be done not
            #  in mini-batches)
            for batch_idx in range(validation_num_batches):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x_validation[:, start_idx:end_idx]
                batchY = y_validation[:, start_idx:end_idx]

                validation_batch_cost, current_validation_state =\
                    myRnn.predict(batchX,
                                  batchY,
                                  current_validation_state,
                                  1.0)
                cumul_loss = cumul_loss + validation_batch_cost

            valid_preplixity = cumul_loss/validation_num_batches/truncated_backprop_length
            logging.info('validation data loss = %f',valid_preplixity)
            if prev_valid_preplixity!=None and valid_preplixity - prev_valid_preplixity > 0:
                myRnn.learning_rate = myRnn.learning_rate / 2
                decay = decay + 1
                logging.info("loading epoch %d parameters, preplixity %f", best_epoch,best_valid_preplixity)
                myRnn._saver.restore(sess,model_path+'_epoch%d.ckpt' % best_epoch)

            prev_valid_preplixity = valid_preplixity
            valid_preplixities.append(valid_preplixity)
            if valid_preplixity <= np.min(valid_preplixities):
                best_epoch = epoch_idx
                best_valid_preplixity = valid_preplixity
                save_path = myRnn._saver.save(sess,model_path+'_epoch%d.ckpt' % epoch_idx)
                logging.info('saved model to file: %s',save_path)

            if decay > max_learning_rate_decay:
                logging.info("reached maximum decay number, quitting after epoch%d",epoch_idx)
                break
    return best_valid_preplixity,best_epoch



# this function is written to do the testing after the training is completely done
# further we can manually stop the training at any time and use the so-far trained model
# stored in model_path for evaluation on the test set
def test_with_batches(test_inputs, test_labels,best_epoch):
    cumul_loss = 0
    x_test = test_inputs
    y_test = test_labels

    tf.reset_default_graph()

    with tf.Session() as sess:
        myRnn = SmoothedSoftmax(session=sess,
                                learning_rate=learning_rate,
                                state_size=state_size,
                                num_classes=vocab_size,
                                num_layers=num_layers,
                                truncated_backprop_length=test_truncated_length,
                                embed_dim=embedding_size,
                                batch_size=test_batch_size,
                                beta=beta,
                                smoothing=smoothing)
        logging.info("loading epoch %d parameters", best_epoch)
        myRnn._saver.restore(sess, model_path + '_epoch%d.ckpt' % best_epoch)

        current_test_state = sess.run(myRnn._initial_state)  # initial state of multi-layer LSTM
        # calculate the loss on the test data in mini-batches, it can also be done not in batches
        prev_loss = cumul_loss
        cumul_loss = 0
        # extracting batches
        for batch_idx in range(test_num_batches):
            start_idx = batch_idx * test_truncated_length
            end_idx = start_idx + test_truncated_length

            batchX = x_test[:, start_idx:end_idx]
            batchY = y_test[:, start_idx:end_idx]

            batch_cost, current_test_state = \
                myRnn.predict(batchX, batchY, current_test_state, 1)
            cumul_loss = cumul_loss + batch_cost
        logging.info("testing loss = %f",
                     cumul_loss / test_num_batches / test_truncated_length)


logging.basicConfig(filename=log_filename,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logging.info("model is : %s", model_filename)
logging.info("extracting train and test inputs and labels as a very long sequence of words")
x_train, y_train, x_valid, y_valid,\
x_tes, y_tes, train_length, validation_length, test_length = preProcess(
    train_filename,
    validation_filename,
    test_filename,
    vocab_size,
    batch_size)

num_batches = train_length // batch_size // truncated_backprop_length
test_num_batches = test_length // batch_size // truncated_backprop_length
validation_num_batches = validation_length // batch_size // truncated_backprop_length
best_epoch = 0
logging.info("Training begins:")
_,best_epoch = train_with_batches(x_train, y_train, x_valid, y_valid)

_, _, _, _, x_tes, y_tes, _, _, test_length = preProcess(train_filename,
                                                        validation_filename,
                                                        test_filename,
                                                        vocab_size,
                                                        test_batch_size)
test_num_batches = test_length // test_batch_size // test_truncated_length
logging.info("Testing begins:")
test_with_batches(x_tes, y_tes,best_epoch)
