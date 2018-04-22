## Experiments for using Neural Nets for Unigrams,
# and check where the output equals empirical distribution or not
import tensorflow as tf
import numpy as np
import time
start_time=time.time()

#implements the gated-version of Kneser-Ney using Neural Nets
class sillyModel():
    def __init__(self,
                 session=None,
                 batch_size=20,
                 learning_rate=0.05,
                 hidden_dim1=256,
                 hidden_dim2=128,
                 num_classes=1000,
                 regularizer=0.0):
        self.sess = session
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes
        self.regularizer = regularizer
        self._init_params()
        self._init_variables()

        self.inference_ops()
        self.loss_ops()
        self.optimization_ops()


        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self._saver = tf.train.Saver()

    def _init_params(self):

        self.y = tf.placeholder(tf.int32, [None],"next_word")
        self.keep_prob = tf.placeholder(tf.float32)

    def _init_variables(self):

        # self.W = tf.Variable(tf.truncated_normal([1,self.hidden_dim1],
        #                                          stddev=0.01))
        # self.U = tf.Variable(tf.truncated_normal([self.hidden_dim1, self.num_classes],
        #                                          stddev=0.0001))
        self.W = tf.Variable(tf.truncated_normal([1,self.num_classes],stddev=0.0001))



    def inference_ops(self):

        # self.h = tf.nn.relu(self.W)
        # self.logits = tf.matmul(self.h,self.U)
        # self.logits = tf.nn.dropout(self.logits, keep_prob=self.keep_prob)
        # self.pred = tf.nn.softmax(self.logits)
        # self.logits = tf.tile(self.logits,[self.batch_size,1])

        self.logits = self.W
        self.pred = tf.nn.softmax(self.logits)
        self.logits = tf.tile(self.logits,[self.batch_size,1])
        print('### logits:', self.logits.get_shape().as_list())

        return

    def loss_ops(self):
        self.weights = tf.trainable_variables()
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.weights]) * self.regularizer
        # one_hot_labels = tf.one_hot(indices=tf.cast(self.y,tf.int32),depth=self.num_classes)

        # self.loss = tf.reduce_mean(-tf.reduce_sum(one_hot_labels *
        #                                          tf.log(self.logits),reduction_indices=[1]))
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,
                                                                                 logits=self.logits))
        return

    def optimization_ops(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss+self.l2_loss)
        capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs]
        self.train_step= optimizer.apply_gradients(capped_gvs)
        return

    def batch_optimization(self, labels, keep_prob):
        feed_dict = { self.y: labels,self.keep_prob:keep_prob}
        batch_cost, _, pred= self.sess.run([self.loss, self.train_step,self.pred],
                                                feed_dict=feed_dict)
        return batch_cost, pred

    def predict(self,labels,keep_prob):
        feed_dict = {self.y: labels, self.keep_prob:keep_prob}
        batch_cost, predictions = self.sess.run([self.loss, self.pred], feed_dict=feed_dict)
        return batch_cost, predictions



##### training parameters ##################
passes = 100
batch_size = 20  #for training
test_batch_size = 20
train_len = 10000
test_len = 10000
learning_rate = 0.01
vocab_size = 1000
hidden_dim1 = 128
hidden_dim2 = 256
classes_size = vocab_size
print_frequency = 100
model_filename = './saved_models/SillyExp'  # path for final trained model


######## Generating data according to a power-law ##############
# alpha = 1.3
# p = [1.0/(i+1)**alpha for i in range(vocab_size)]
# s = sum(p)
# p = [x/s for x in p]
# x = [0 for _ in range(train_len+test_len)]
# for i in range(train_len + test_len):
#     x[i] = np.random.choice(np.arange(0, vocab_size),p=p)
# x_train = x[:train_len]
# x_test = x[train_len:]
# np.savez('../data/syn/power-law',x_train=x_train,x_test=x_test)
files = np.load('../data/syn/power-law.npz')
x_train = files['x_train']
x_test = files ['x_test']

emp,_ = np.histogram(x_train, bins=range(vocab_size))
emp = emp * 1.0 / train_len
def print_time():
    print '--- time: ', (time.time() - start_time) // 60, \
        'minutes', (time.time() - start_time) % 60, 'seconds'



def train_with_minibatch(train_data, test_data, model_filename, passes, batch_size,test_batch_size):
    model_path = model_filename
    with tf.Session() as sess:

        my_model = sillyModel(session=sess,
                              batch_size=batch_size,
                              learning_rate=learning_rate,
                              hidden_dim1=hidden_dim1,
                              hidden_dim2=hidden_dim2,
                              num_classes=classes_size)

        for pass_number in range(passes):
            print 'pass#: ', pass_number + 1
            total_train_loss = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                labels = train_data[start_idx:end_idx]
                batch_cost,pred = my_model.batch_optimization(labels=labels,
                                                         keep_prob=1.0)
                total_train_loss = total_train_loss + batch_cost

                if batch_idx % print_frequency == 0:
                    print 'training loss of batch', batch_idx, ': = ', batch_cost

            cumul_loss = 0
            for i in range(test_num_batches):
                start_idx = i * test_batch_size
                end_idx = start_idx + test_batch_size
                labels = test_data[start_idx:end_idx]

                batch_cost, _ = my_model.predict(labels=labels,
                                                 keep_prob=1)
                cumul_loss = cumul_loss + batch_cost
            print 'training loss after batch', pass_number+1, ': = ', total_train_loss / num_batches
            print 'testing loss after pass', pass_number+1, ': = ', cumul_loss / test_num_batches

            if model_path:
                my_model._saver.save(sess, model_path)

    return pred

def predict(test_data, model_path):

    tf.reset_default_graph()

    with tf.Session() as sess:

        my_model = sillyModel(session=sess,
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
            labels = test_data[start_idx:end_idx]
            batch_cost,predictions = my_model.predict(labels=labels,
                                                      keep_prob=1)
            cumul_loss = cumul_loss + batch_cost
        print  'test cross entropy:', cumul_loss/test_num_batches

        #########################################################

train_data = x_train
test_data = x_test
num_batches = train_len // batch_size
test_num_batches = test_len // batch_size

print 'Training using Neural Networks'
pred = train_with_minibatch(train_data,test_data,model_filename, passes, batch_size,test_batch_size)
# plt.plot(np.reshape(np.array(range(vocab_size)),[1000,1]),pred)
# plt.show()
print '---- time: ', print_time()
print 'Testing on test data...'
predict(test_data, model_filename)
print '---- time: ', print_time()
