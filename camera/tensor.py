import tensorflow as tf
import numpy as np
import math

class BasicTensor(object):
    def __init__(self, labels, image_size, check_point, model_name="model"):
        self.labels_d = labels
        self.labels_i = {v: k for k, v in self.labels_d.items()}
        self.num_labels = len(self.labels_d)
        self.image_size = image_size
        self.check_point = check_point
        self.model_name = model_name
        
    def position_index(self, label):
        for i, e in enumerate(label):
            if e == 1:
                return i

    def convert_label(self, label):
        #[0, 0, 1.0] -> 155
        try:
            return self.labels_d[self.position_index(label)]
        except KeyError:
            return None
            #max_value = max(label)
            #for i, e in enumerate(label):
            #    if e == max_value:
            #        return self.labels_d[i]

    # batch_size has to be less than the len of training set label
    def fit(self, test_dataset, valid_dataset, batch_size):
        # With gradient descent training, even this much data is prohibitive.
        # Subset the training data for faster turnaround.
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            self.tf_train_dataset = tf.placeholder(tf.float32,
                                            shape=(batch_size, self.image_size * self.image_size))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, self.num_labels))
            self.tf_valid_dataset = tf.constant(valid_dataset)
            self.tf_test_dataset = tf.constant(test_dataset)

            # Variables.
            # These are the parameters that we are going to be training. The weight
            # matrix will be initialized using random valued following a (truncated)
            # normal distribution. The biases get initialized to zero.
            weights = tf.Variable(
                tf.truncated_normal([self.image_size * self.image_size, self.num_labels]))
            biases = tf.Variable(tf.zeros([self.num_labels]))

            # Training computation.
            # We multiply the inputs with the weight matrix, and add biases. We compute
            # the softmax and cross-entropy (it's one operation in TensorFlow, because
            # it's very common, and it can be optimized). We take the average of this
            # cross-entropy across all training examples: that's our loss.
            self.logits = tf.matmul(self.tf_train_dataset, weights) + biases
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))

            #regularizers = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
            #self.loss += 5e-4 * regularizers
            # Optimizer.
            # We are going to find the minimum of this loss using gradient descent.
            self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

            # Predictions for the training, validation, and test data.
            # These are not part of training, but merely here so that we can report
            # accuracy figures as we train.
            self.train_prediction = tf.nn.softmax(self.logits)
            self.valid_prediction = tf.nn.softmax(
                tf.matmul(self.tf_valid_dataset, weights) + biases)
            self.test_prediction = tf.nn.softmax(tf.matmul(self.tf_test_dataset, weights) + biases)

    def score(self, train_dataset, train_labels, test_labels, valid_labels, batch_size):
        num_steps = 2001
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()
            print("Initialized")
            for step in xrange(num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                # Generate a minibatch.
                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
                _, l, predictions = session.run(
                  [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (step % 500 == 0):
                    print("Minibatch loss at step: {}, {}".format(step, l))
                    print("Minibatch accuracy: %.1f" % self.accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f" % self.accuracy(
                        self.valid_prediction.eval(), valid_labels))
            score_v = self.accuracy(self.test_prediction.eval(), test_labels)
            print('Test accuracy: %.1f' % score_v)
            saver.save(session, '{}{}.ckpt'.format(self.check_point, self.model_name), global_step=step)
            return score_v

    def predict(self, img):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.check_point)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("...no checkpoint found...")

            feed_dict = {self.tf_train_dataset: img}
            classification = session.run(self.train_prediction, feed_dict=feed_dict)
            print(classification)
            return self.convert_label(classification[0])

    def accuracy(self, predictions, labels):
        return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])

class TowLayerTensor(BasicTensor):
    def layers(self):
        size = 10
        W1 = tf.Variable(
            tf.truncated_normal([self.image_size * self.image_size, size]), name='weights')
        b1 = tf.Variable(tf.zeros([size]), name='biases')
        hidden = tf.nn.relu(tf.matmul(self.tf_train_dataset, W1) + b1)

        W2 = tf.Variable(
            tf.truncated_normal([size, self.num_labels]))
        b2 = tf.Variable(tf.zeros([self.num_labels]))

        hidden = tf.nn.dropout(hidden, 0.5, seed=66478)
        self.logits = tf.matmul(hidden, W2) + b2
        return W1, b1, W2, b2

    def fit(self, test_dataset, valid_dataset, batch_size):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_train_dataset = tf.placeholder(tf.float32,
                                            shape=(batch_size, self.image_size * self.image_size))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, self.num_labels))
            self.tf_valid_dataset = tf.constant(valid_dataset)
            self.tf_test_dataset = tf.constant(test_dataset)

            W1, b1, W2, b2 = self.layers()

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))

            regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2)
            self.loss += 5e-4 * regularizers

            self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

            self.train_prediction = tf.nn.softmax(self.logits)
            hidden_valid =  tf.nn.relu(tf.matmul(self.tf_valid_dataset, W1) + b1)
            valid_logits = tf.matmul(hidden_valid, W2) + b2
            self.valid_prediction = tf.nn.softmax(valid_logits)
            hidden_test = tf.nn.relu(tf.matmul(self.tf_test_dataset, W1) + b1)
            test_logits = tf.matmul(hidden_test, W2) + b2
            self.test_prediction = tf.nn.softmax(test_logits)

class ConvTensor(BasicTensor):
    def model(self, data, layer1_weights, layer1_biases, layer2_weights, layer2_biases, 
            layer3_weights, layer3_biases, dropout=False):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        pool = tf.nn.max_pool(hidden,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer2_weights) + layer2_biases)
        if dropout:
            hidden = tf.nn.dropout(hidden, 0.5, seed=66478)
        return tf.matmul(hidden, layer3_weights) + layer3_biases

    def fit(self, test_dataset, valid_dataset, batch_size, patch_size, depth, num_channels, num_hidden):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_train_dataset = tf.placeholder(
                tf.float32, shape=(batch_size, self.image_size, self.image_size, num_channels))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, self.num_labels))
            self.tf_valid_dataset = tf.constant(valid_dataset)
            self.tf_test_dataset = tf.constant(test_dataset)

            # Variables.
            layer3_size = int(math.ceil(self.image_size / 4.))
            layer1_weights = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, num_channels, depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([depth]))
            layer2_weights = tf.Variable(tf.truncated_normal(
                [layer3_size * layer3_size * depth, num_hidden], stddev=0.1)) # 4 num of ksize
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
            layer3_weights = tf.Variable(tf.truncated_normal(
                [num_hidden, self.num_labels], stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_labels]))

            self.logits = self.model(self.tf_train_dataset, layer1_weights, 
                layer1_biases, layer2_weights, layer2_biases, layer3_weights, 
                layer3_biases, dropout=True)

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))
            regularizers = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer1_biases) +\
            tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_biases) +\
            tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases)
            self.loss += 5e-4 * regularizers

            # Optimizer: set up a variable that's incremented once per batch and
            # controls the learning rate decay.
            batch = tf.Variable(0)
            # Decay once per epoch, using an exponential schedule starting at 0.01.
            learning_rate = tf.train.exponential_decay(
              0.01,                # Base learning rate.
              batch * batch_size,  # Current index into the dataset.
              23,          # train_labels.shape[0] Decay step.
              0.95,                # Decay rate.
              staircase=True)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss,
                global_step=batch)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(self.logits)
            self.valid_prediction = tf.nn.softmax(self.model(self.tf_valid_dataset, layer1_weights, 
                layer1_biases, layer2_weights, layer2_biases, layer3_weights, 
                layer3_biases))
            self.test_prediction = tf.nn.softmax(self.model(self.tf_test_dataset, layer1_weights, 
                layer1_biases, layer2_weights, layer2_biases, layer3_weights, 
                layer3_biases))

    def score(self, train_dataset, train_labels, test_labels, valid_labels, batch_size):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()
            print("Initialized")
            for step in xrange(int(150 * train_labels.shape[0]) // batch_size):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
                _, l, predictions = session.run(
                [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (step % 5000 == 0):
                    print "Minibatch loss at step", step, ":", l
                    print "Minibatch accuracy: %.1f%%" % self.accuracy(predictions, batch_labels)
                    print "Validation accuracy: %.1f%%" % self.accuracy(
                    self.valid_prediction.eval(), valid_labels)
            score_v = self.accuracy(self.test_prediction.eval(), test_labels)
            print('Test accuracy: %.1f' % score_v)
            saver.save(session, '{}{}.ckpt'.format(self.check_point, self.model_name), global_step=step)
            return score_v
