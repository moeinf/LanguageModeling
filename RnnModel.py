import tensorflow as tf
from utils import *
import operator

#Basic RNN model, not using Tensorflow Gradient descent, from Denny Britz tutorial on WildML
class BasicRNN():
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is too large fail the gradient check
                if relative_error >= error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)

    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

#Basic RNN model, not using TensorFlow cell
class cellRNN():
    def __init__(self,
                 session=None,
                 learning_rate=0.005,
                 state_size=4,
                 num_classes=2,
                 truncated_backprop_length=3,
                 batch_size=20,
                 series_length=50000):
        self.sess = session
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.truncated_backprop_length = truncated_backprop_length
        self.batch_size = batch_size
        self.series_length = series_length
        self.state_size = state_size

        self._init_params()
        self._init_variables()

        self.inference_ops()
        self.loss_ops()
        self.optimization_ops()
        self.predict_ops()

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self._saver = tf.train.Saver()


    def _init_params(self):
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.truncated_backprop_length])
        self.y = tf.placeholder(tf.int32,[self.batch_size,self.truncated_backprop_length])
        self.initial_state = tf.placeholder(tf.float32,[self.batch_size,self.state_size])

    def _init_variables(self):

        self.W = tf.Variable(np.random.rand(self.state_size+1,self.state_size),dtype=tf.float32)
        self.b = tf.Variable(np.zeros((1, self.state_size)), dtype=tf.float32)

        self.W2 = tf.Variable(np.random.rand(self.state_size, self.num_classes), dtype=tf.float32)
        self.b2 = tf.Variable(np.zeros((1, self.num_classes)), dtype=tf.float32)



    def inference_ops(self):

        self.inputs_series = tf.unstack(self.x, axis=1)
        self.labels_series = tf.unstack(self.y, axis=1)
        current_state = self.initial_state
        # Forward pass
        states_series = []
        for current_input in self.inputs_series:
            current_input = tf.reshape(current_input, [self.batch_size, 1])
            input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns

            next_state = tf.tanh(tf.matmul(input_and_state_concatenated, self.W) + self.b)  # Broadcasted addition
            #print next_state.shape
            states_series.append(next_state)
            current_state = next_state

        self.final_state = current_state
        self.logits_series = [tf.matmul(state, self.W2) + self.b2 for state in states_series]  # Broadcasted addition
        #print self.logits_series[-1].shape



    def loss_ops(self):
        self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
                       for logits, labels in zip(self.logits_series, self.labels_series)]
        self.total_loss = tf.reduce_mean(self.losses)
    def optimization_ops(self):
        self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss)

    def predict_ops(self):
        self.predictions_series = [tf.nn.softmax(logits) for logits in self.logits_series]
        self.single_prediction = [tf.argmax(logits,1) for logits in self.logits_series]

    def batch_optimiztion(self, inputs, labels, state):
        feed_dict = {self.x: inputs, self.y:labels, self.initial_state:state}

        batch_cost, __ , state = self.sess.run([self.total_loss, self.train_step, self.final_state],
                                       feed_dict=feed_dict)


        return batch_cost, state

    def predict(self,inputs,labels, state):
        feed_dict = {self.x: inputs, self.y: labels, self.initial_state:state}

        predictions, single_predictions, batch_cost,state = self.sess.run([self.predictions_series,
                                                              self.single_prediction,
                                                         self.total_loss,self.final_state],feed_dict=feed_dict)

        return predictions,single_predictions, batch_cost,state

#Basic RNN model, using TensorFlow cell
class TfCellRNN():
    def __init__(self,
                 session=None,
                 learning_rate=0.005,
                 state_size=4,
                 num_classes=2,
                 truncated_backprop_length=3,
                 batch_size=20,
                 series_length=50000):
        self.sess = session
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.truncated_backprop_length = truncated_backprop_length
        self.batch_size = batch_size
        self.series_length = series_length
        self.state_size = state_size

        self._init_params()
        self._init_variables()

        self.inference_ops()
        self.loss_ops()
        self.optimization_ops()
        self.predict_ops()

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self._saver = tf.train.Saver()


    def _init_params(self):
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.truncated_backprop_length])
        self.y = tf.placeholder(tf.int32,[self.batch_size,self.truncated_backprop_length])
        self.initial_state = tf.placeholder(tf.float32,[self.batch_size,self.state_size])

    def _init_variables(self):


        self.W2 = tf.Variable(np.random.rand(self.state_size, self.num_classes), dtype=tf.float32)
        self.b2 = tf.Variable(np.zeros((1, self.num_classes)), dtype=tf.float32)



    def inference_ops(self):

        self.inputs_series = tf.split(self.x, self.truncated_backprop_length, axis=1)
        #self.inputs_series = tf.unstack(self.x,axis=1)
        self.labels_series = tf.unstack(self.y, axis=1)
        # current_state = self.initial_state
        # Forward pass
        self.cell = tf.contrib.rnn.BasicRNNCell(self.state_size)
        states_series, self.final_state = tf.contrib.rnn.static_rnn(self.cell, self.inputs_series, self.initial_state)
        self.logits_series = [tf.matmul(state, self.W2) + self.b2 for state in states_series]  # Broadcasted addition




    def loss_ops(self):
        self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
                       for logits, labels in zip(self.logits_series, self.labels_series)]
        self.total_loss = tf.reduce_mean(self.losses)
    def optimization_ops(self):
        self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss)

    def predict_ops(self):
        self.predictions_series = [tf.nn.softmax(logits) for logits in self.logits_series]
        self.single_prediction = [tf.argmax(logits,1) for logits in self.logits_series]

    def batch_optimiztion(self, inputs, labels, state):
        feed_dict = {self.x: inputs, self.y:labels, self.initial_state:state}

        batch_cost, __ , state = self.sess.run([self.total_loss, self.train_step, self.final_state],
                                       feed_dict=feed_dict)


        return batch_cost, state

    def predict(self,inputs,labels, state):
        feed_dict = {self.x: inputs, self.y: labels, self.initial_state:state}

        predictions, single_predictions, batch_cost,state = self.sess.run([self.predictions_series,
                                                              self.single_prediction,
                                                         self.total_loss,self.final_state],feed_dict=feed_dict)

        return predictions,single_predictions, batch_cost,state

#Single-layer LSTM model, using TensorFlow LSTM cell
class TfCellLSTM():
    def __init__(self,
                 session=None,
                 learning_rate=0.005,
                 state_size=4,
                 num_classes=2,
                 truncated_backprop_length=3,
                 batch_size=20,
                 series_length=50000):
        self.sess = session
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.truncated_backprop_length = truncated_backprop_length
        self.batch_size = batch_size
        self.series_length = series_length
        self.state_size = state_size

        self._init_params()
        self._init_variables()

        self.inference_ops()
        self.loss_ops()
        self.optimization_ops()
        self.predict_ops()

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self._saver = tf.train.Saver()


    def _init_params(self):
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.truncated_backprop_length])
        self.y = tf.placeholder(tf.int32,[self.batch_size,self.truncated_backprop_length])

        self.cell_state = tf.placeholder(tf.float32, [self.batch_size, self.state_size])
        self.hidden_state = tf.placeholder(tf.float32, [self.batch_size, self.state_size])
        self.initial_state = tf.contrib.rnn.LSTMStateTuple(self.cell_state, self.hidden_state)

    def _init_variables(self):


        self.W2 = tf.Variable(np.random.rand(self.state_size, self.num_classes), dtype=tf.float32)
        self.b2 = tf.Variable(np.zeros((1, self.num_classes)), dtype=tf.float32)



    def inference_ops(self):

        self.inputs_series = tf.split(self.x, self.truncated_backprop_length, axis=1)
        #self.inputs_series = tf.unstack(self.x,axis=1)
        self.labels_series = tf.unstack(self.y, axis=1)
        # current_state = self.initial_state
        # Forward pass
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True)
        states_series, self.final_state = tf.contrib.rnn.static_rnn(self.cell, self.inputs_series, self.initial_state)
        # self.cell_state, self.hidden_state = self.final_state
        self.logits_series = [tf.matmul(state, self.W2) + self.b2 for state in states_series]  # Broadcasted addition


    def loss_ops(self):
        self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
                       for logits, labels in zip(self.logits_series, self.labels_series)]
        self.total_loss = tf.reduce_mean(self.losses)
    def optimization_ops(self):
        self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss)

    def predict_ops(self):
        self.predictions_series = [tf.nn.softmax(logits) for logits in self.logits_series]
        self.single_prediction = [tf.argmax(logits,1) for logits in self.logits_series]

    def batch_optimiztion(self, inputs, labels, cell_state, hidden_state):
        feed_dict = {self.x: inputs, self.y:labels, self.cell_state: cell_state, self.hidden_state: hidden_state}

        batch_cost, __ , current_state = self.sess.run([self.total_loss, self.train_step, self.final_state],
                                       feed_dict=feed_dict)

        cell_state, hidden_state = current_state
        return batch_cost, cell_state, hidden_state

    def predict(self,inputs,labels, cell_state, hidden_state):
        feed_dict = {self.x: inputs, self.y: labels, self.cell_state:cell_state, self.hidden_state: hidden_state}

        predictions, single_predictions, batch_cost,state = self.sess.run([self.predictions_series,
                                                              self.single_prediction,
                                                         self.total_loss,self.final_state],feed_dict=feed_dict)

        return predictions,single_predictions, batch_cost,state

#GRU model, not using TensorFlow GRU cell
class cellGRU():
    def __init__(self,
                 session=None,
                 learning_rate=0.005,
                 state_size=4,
                 num_classes=2,
                 truncated_backprop_length=3,
                 batch_size=20,
                 series_length=50000):
        self.sess = session
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.truncated_backprop_length = truncated_backprop_length
        self.batch_size = batch_size
        self.series_length = series_length
        self.state_size = state_size

        self._init_params()
        self._init_variables()

        self.inference_ops()
        self.loss_ops()
        self.optimization_ops()
        self.predict_ops()

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self._saver = tf.train.Saver()


    def _init_params(self):
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.truncated_backprop_length])
        self.y = tf.placeholder(tf.int32,[self.batch_size,self.truncated_backprop_length])
        self.initial_state = tf.placeholder(tf.float32,[self.batch_size,self.state_size])

    def _init_variables(self):

        self.W = tf.Variable(np.random.rand(self.state_size+1,self.state_size),dtype=tf.float32)
        self.b = tf.Variable(np.zeros((1, self.state_size)), dtype=tf.float32)

        self.W_r = tf.Variable(np.random.rand(self.state_size+1,self.state_size),dtype=tf.float32)
        self.b_r = tf.Variable(np.zeros((1,self.state_size)),dtype=tf.float32)

        self.W_z = tf.Variable(np.random.rand(self.state_size + 1, self.state_size), dtype=tf.float32)
        self.b_z = tf.Variable(np.zeros((1, self.state_size)), dtype=tf.float32)

        self.W2 = tf.Variable(np.random.rand(self.state_size, self.num_classes), dtype=tf.float32)
        self.b2 = tf.Variable(np.zeros((1, self.num_classes)), dtype=tf.float32)






    def inference_ops(self):

        self.inputs_series = tf.unstack(self.x, axis=1)
        self.labels_series = tf.unstack(self.y, axis=1)
        current_state = self.initial_state
        # Forward pass
        states_series = []
        for current_input in self.inputs_series:
            current_input = tf.reshape(current_input, [self.batch_size, 1])
            input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns
            r = tf.sigmoid(tf.matmul(input_and_state_concatenated,self.W_r)+self.b_r)
            z = tf.sigmoid(tf.matmul(input_and_state_concatenated,self.W_z)+self.b_z)
            modified_input_and_state_concatenated = tf.concat([current_input,tf.multiply(r, current_state)],1)
            candiate_next_state = tf.tanh(tf.matmul(modified_input_and_state_concatenated, self.W) + self.b)  # Broadcasted addition
            ztilde = tf.add(tf.multiply(tf.constant(-1,tf.float32,[self.batch_size,self.state_size]),z),
                            tf.ones([self.batch_size,self.state_size]))
            next_state = tf.add(tf.multiply(z,current_state),tf.multiply(ztilde,candiate_next_state))
            #print next_state.shape
            states_series.append(next_state)
            current_state = next_state

        self.final_state = current_state
        self.logits_series = [tf.matmul(state, self.W2) + self.b2 for state in states_series]  # Broadcasted addition
        #print self.logits_series[-1].shape



    def loss_ops(self):
        self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
                       for logits, labels in zip(self.logits_series, self.labels_series)]
        self.total_loss = tf.reduce_mean(self.losses)
    def optimization_ops(self):
        self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss)

    def predict_ops(self):
        self.predictions_series = [tf.nn.softmax(logits) for logits in self.logits_series]
        self.single_prediction = [tf.argmax(logits,1) for logits in self.logits_series]

    def batch_optimiztion(self, inputs, labels, state):
        feed_dict = {self.x: inputs, self.y:labels, self.initial_state:state}

        batch_cost, __ , state = self.sess.run([self.total_loss, self.train_step, self.final_state],
                                       feed_dict=feed_dict)


        return batch_cost, state

    def predict(self,inputs,labels, state):
        feed_dict = {self.x: inputs, self.y: labels, self.initial_state:state}

        predictions, single_predictions, batch_cost,state = self.sess.run([self.predictions_series,
                                                              self.single_prediction,
                                                         self.total_loss,self.final_state],feed_dict=feed_dict)

        return predictions,single_predictions, batch_cost,state

#Multi-layer LSTM, using TensorFlow LSTM cell
class TfMultiCellLSTM():
    def __init__(self,
                 session=None,
                 learning_rate=0.005,
                 state_size=4,
                 num_classes=2,
                 num_layers=3,
                 truncated_backprop_length=3,
                 batch_size=20,
                 series_length=50000):
        self.sess = session
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.truncated_backprop_length = truncated_backprop_length
        self.batch_size = batch_size
        self.series_length = series_length
        self.state_size = state_size

        self._init_params()
        self._init_variables()

        self.inference_ops()
        self.loss_ops()
        self.optimization_ops()
        self.predict_ops()

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self._saver = tf.train.Saver()


    def _init_params(self):
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.truncated_backprop_length])
        self.y = tf.placeholder(tf.int32,[self.batch_size,self.truncated_backprop_length])
        self.initial_state = tf.placeholder(tf.float32, [self.num_layers, 2, self.batch_size, self.state_size])

    def _init_variables(self):
        self.W2 = tf.Variable(np.random.rand(self.state_size, self.num_classes), dtype=tf.float32)
        self.b2 = tf.Variable(np.zeros((1, self.num_classes)), dtype=tf.float32)

    def inference_ops(self):
        #unpacking the input and states to the desired format
        self.inputs_series = tf.split(self.x, self.truncated_backprop_length, axis=1)
        self.labels_series = tf.unstack(self.y, axis=1)
        state_per_layer_list = tf.unstack(self.initial_state, axis=0)
        rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(self.num_layers)])
        # Forward pass
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True)
        self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * self.num_layers, state_is_tuple=True)
        states_series, self.final_state = tf.contrib.rnn.static_rnn(self.cell, self.inputs_series, rnn_tuple_state)
        self.logits_series = [tf.matmul(state, self.W2) + self.b2 for state in states_series]


    def loss_ops(self):
        self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
                       for logits, labels in zip(self.logits_series, self.labels_series)]
        self.total_loss = tf.reduce_mean(self.losses)
    def optimization_ops(self):
        self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss)

    def predict_ops(self):
        self.predictions_series = [tf.nn.softmax(logits) for logits in self.logits_series]
        self.single_prediction = [tf.argmax(logits,1) for logits in self.logits_series]

    def batch_optimiztion(self, inputs, labels, state):
        feed_dict = {self.x: inputs, self.y:labels, self.initial_state:state}

        batch_cost, __ , current_state = self.sess.run([self.total_loss, self.train_step, self.final_state],
                                       feed_dict=feed_dict)
        return batch_cost, current_state

    def predict(self,inputs,labels, state):
        feed_dict = {self.x: inputs, self.y: labels, self.initial_state:state}
        predictions, single_predictions, batch_cost,state = self.sess.run([self.predictions_series,
                                                              self.single_prediction,
                                                         self.total_loss,self.final_state],feed_dict=feed_dict)
        return predictions,single_predictions, batch_cost,state
