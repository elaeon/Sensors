import tensorflow as tf
import numpy as np

class BasicTensor(object):
    def __init__(self, labels, image_size, check_point):
        self.labels_d = labels
        self.labels_i = {v: k for k, v in self.labels_d.items()}
        self.num_labels = len(self.labels_d)
        self.image_size = image_size
        self.check_point = check_point
        
    def position_index(self, label):
        for i, e in enumerate(label):
            if e == 1:
                return i

    def convert_label(self, label):
        #[0, 0, 1.0] -> 155
        return self.labels_d[self.position_index(label)]

    # batch_size has to be less than the len of training set label
    def fit(self, test_dataset, valid_dataset, batch_size):
        import tensorflow as tf
        # With gradient descent training, even this much data is prohibitive.
        # Subset the training data for faster turnaround.
        self.graph = tf.Graph()
        with graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            tf_train_dataset = tf.placeholder(tf.float32,
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

            # Optimizer.
            # We are going to find the minimum of this loss using gradient descent.
            self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

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
                offset = (step * batch_size) % (self.train_labels.shape[0] - batch_size)
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
                    print("Minibatch loss at step", step, ":", l)
                    print("Minibatch accuracy: %.1f" % self.accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f" % self.accuracy(
                        self.valid_prediction.eval(), valid_labels))
            score_v = self.accuracy(self.test_prediction.eval(), test_labels)
            print('Test accuracy: %.1f' % score_v)
            saver.save(session, 'model.ckpt', global_step=step)
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
            return self.convert_label(classification[0])

    def accuracy(self, predictions, labels):
        return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])
