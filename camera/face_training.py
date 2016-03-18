import os
import numpy as np
from skimage import io as sio
from skimage import color
from skimage import filters
from skimage import transform
from sklearn import preprocessing


class BasicFaceClassif(object):
    def __init__(self, model=None):
        self.image_size = 90
        self.folder = "/home/sc/Pictures/face/"
        np.random.seed(133)
        if model is None:
            self.model = None
        else:
            self.load(model)

    def load_images(self):
        images = os.listdir(self.folder)
        max_num_images = len(images)
        dataset = np.ndarray(
            shape=(max_num_images, self.image_size, self.image_size), dtype=np.float32)
        labels = np.ndarray(shape=(max_num_images), dtype=np.float32)
        #min_max_scaler = preprocessing.MinMaxScaler()
        for image_index, image in enumerate(images):
            image_file = os.path.join(self.folder, image)
            image_data = sio.imread(image_file)
            number_id = image.split("-")[1]
            if image_data.shape != (self.image_size, self.image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            image_data = image_data.astype(float)
            dataset[image_index, :, :] = preprocessing.scale(image_data)#min_max_scaler.fit_transform(image_data)
            labels[image_index] = number_id
        num_images = image_index
        print 'Full dataset tensor:', dataset.shape
        print 'Mean:', np.mean(dataset)
        print 'Standard deviation:', np.std(dataset)
        print 'Labels:', labels.shape

        return dataset, labels

    def randomize(self, dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    def fit(self):
        pass

    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
        return dataset, labels

    def set_dataset(self, valid_size_p=.1, train_size_p=.6):
        dataset, labels = self.load_images()
        total_size = dataset.shape[0]
        valid_size = total_size * valid_size_p
        train_size = total_size * train_size_p
        v_t_size = valid_size + train_size
        self.train_dataset, self.train_labels = self.randomize(dataset[:v_t_size], labels[:v_t_size])
        self.valid_dataset = self.train_dataset[:valid_size,:,:]
        self.valid_labels = self.train_labels[:valid_size]
        self.train_dataset = self.train_dataset[valid_size:v_t_size,:,:]
        self.train_labels = self.train_labels[valid_size:v_t_size]
        self.test_dataset, self.test_labels = self.randomize(
            dataset[v_t_size:total_size], labels[v_t_size:total_size])
        self.train_dataset, self.train_labels = self.reformat(self.train_dataset, self.train_labels)
        self.valid_dataset, self.valid_labels = self.reformat(self.valid_dataset, self.valid_labels)
        self.test_dataset, self.test_labels = self.reformat(self.test_dataset, self.test_labels)
        print("Test set: {}, Valid set: {}, Training set: {}".format(
            self.test_labels.shape[0], self.valid_labels.shape[0], self.train_labels.shape[0]))

    def run(self):
        score = self.fit(
            self.train_dataset, self.train_labels, self.test_dataset, 
            self.test_labels, self.valid_dataset, self.valid_labels)
        print("Score: {}%".format((score * 100)))

    def save(self, filename):
        from sklearn.externals import joblib
        joblib.dump(self.model, '{}.pkl'.format(filename)) 

    def load(self, filename):
        from sklearn.externals import joblib
        self.model = joblib.load('{}.pkl'.format(filename))

    def process_images(self, images):
        for score, image, d, idx in images:
            print(image.shape, score)
            img = image[d.top():d.bottom(), d.left():d.right(), 0:3]
            img_gray = color.rgb2gray(img)
            if (self.image_size, self.image_size) < img_gray.shape or\
                img_gray.shape < (self.image_size, self.image_size):
                img_gray = transform.resize(img_gray, (self.image_size, self.image_size))
                img_gray = filters.gaussian_filter(img_gray, .5)
            yield img_gray

    def predict(self, images):
        if self.model is not None:
            img = list(self.process_images(images))[0]
            sio.imsave("/home/sc/Pictures/face-155-X.png", img)
            img = img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
            return self.model.predict(img)

    def predict_set(self, img):
        img = img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
        return self.model.predict(img)

class SVCFace(BasicFaceClassif):
    def __init__(self, model=None):
        super(SVCFace, self).__init__(model=model)
        self.set_dataset()

    def fit(self, train_dataset, train_labels, test_dataset, test_labels):
        from sklearn.linear_model import LogisticRegression
        from sklearn import svm

        reg = LogisticRegression(penalty='l2')
        #reg = svm.SVC(kernel='rbf')
        #reg = svm.LinearSVC(C=1.0, max_iter=1000)
        reg = reg.fit(train_dataset, train_labels)

        score = reg.score(test_dataset, test_labels)
        self.model = reg
        return score

class TensorFace(BasicFaceClassif):
    def __init__(self):
        super(TensorFace, self).__init__()
        self.labels_d = dict(enumerate(["106", "110", "155"]))
        self.labels_i = {v: k for k, v in self.labels_d.items()}
        self.set_dataset()

    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
        #155 -> [0, 0, 1.0]
        #106 -> [1.0, 0, 0]
        #110 -> [0, 1.0, 0]
        new_labels = np.asarray([self.labels_i[str(int(label))] for label in labels])
        labels_m = (np.arange(len(self.labels_d)) == new_labels[:,None]).astype(np.float32)
        return dataset, labels_m

    def position_index(self, label):
        for i, e in enumerate(label):
            if e == 1:
                return i

    def convert_label(self, label):
        #[0, 0, 1.0] -> 155
        return self.labels_d[self.position_index(label)]

    def fit(self, train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels):
        graph, tf_train_dataset, tf_train_labels, optimizer, loss, train_prediction, valid_prediction, test_prediction = self.prepare_graph()
        return self.train(graph, tf_train_dataset, tf_train_labels, optimizer, loss, train_prediction, valid_prediction, test_prediction)

    def train(self, graph, tf_train_dataset, tf_train_labels, optimizer, loss, train_prediction, valid_prediction, test_prediction):
        import tensorflow as tf
        num_steps = 2001
        img = None
        num_labels = len(self.labels_d)
        batch_size = 10
        with tf.Session(graph=graph) as session:
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()
            print("Initialized")
            for step in xrange(num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * batch_size) % (self.train_labels.shape[0] - batch_size)
                # Generate a minibatch.
                batch_data = self.train_dataset[offset:(offset + batch_size), :]
                batch_labels = self.train_labels[offset:(offset + batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                _, l, predictions = session.run(
                  [optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 500 == 0):
                    print("Minibatch loss at step", step, ":", l)
                    print("Minibatch accuracy: %.1f" % self.accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f" % self.accuracy(
                        valid_prediction.eval(), self.valid_labels))
            score = self.accuracy(test_prediction.eval(), self.test_labels)
            print('Test accuracy: %.1f' % score)
            saver.save(session, 'model.ckpt', global_step=step)
            return score

    def predict(self, graph, img, tf_dataset, prediction):
        import tensorflow as tf
        with tf.Session(graph=graph) as session:
            saver = tf.train.Saver()
            img = img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
            ckpt = tf.train.get_checkpoint_state("/home/sc/git/sensors/camera/")
            #print(ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("...no checkpoint found...")

            feed_dict = {tf_dataset: img}
            classification = session.run(prediction, feed_dict=feed_dict)
            return self.convert_label(classification[0])

    # batch_size has to be less than the len of training set label
    def prepare_graph(self, batch_size=10):
        import tensorflow as tf
        # With gradient descent training, even this much data is prohibitive.
        # Subset the training data for faster turnaround.
        num_labels = len(self.labels_d) 
        graph = tf.Graph()
        with graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            tf_train_dataset = tf.placeholder(tf.float32,
                                            shape=(batch_size, self.image_size * self.image_size))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            tf_valid_dataset = tf.constant(self.valid_dataset)
            tf_test_dataset = tf.constant(self.test_dataset)

            # Variables.
            # These are the parameters that we are going to be training. The weight
            # matrix will be initialized using random valued following a (truncated)
            # normal distribution. The biases get initialized to zero.
            weights = tf.Variable(
                tf.truncated_normal([self.image_size * self.image_size, num_labels]))
            biases = tf.Variable(tf.zeros([num_labels]))

            # Training computation.
            # We multiply the inputs with the weight matrix, and add biases. We compute
            # the softmax and cross-entropy (it's one operation in TensorFlow, because
            # it's very common, and it can be optimized). We take the average of this
            # cross-entropy across all training examples: that's our loss.
            logits = tf.matmul(tf_train_dataset, weights) + biases
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

            # Optimizer.
            # We are going to find the minimum of this loss using gradient descent.
            optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

            # Predictions for the training, validation, and test data.
            # These are not part of training, but merely here so that we can report
            # accuracy figures as we train.
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(
                tf.matmul(tf_valid_dataset, weights) + biases)
            test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

        return graph, tf_train_dataset, tf_train_labels, optimizer, loss, train_prediction, valid_prediction, test_prediction

    def predict_set(self, img):
        graph, tf_train_dataset, tf_train_labels, optimizer, loss, train_prediction, valid_prediction, test_prediction = self.prepare_graph(batch_size=1)
        return self.predict(graph, img, tf_train_dataset, train_prediction)
        
    def predict_v(self, images):
        img = list(self.process_images(images))[0]
        graph, tf_train_dataset, tf_train_labels, optimizer, loss, train_prediction, valid_prediction, test_prediction = self.prepare_graph(batch_size=1)
        #sio.imsave("/home/sc/Pictures/face-155-X.png", img)
        #img = img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
        return self.predict(graph, img, tf_train_dataset, train_prediction)

    def accuracy(self, predictions, labels):
        return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])

if __name__  == '__main__':
    face_classif = TensorFace()
    #face_classif = SVCFace()
    face_classif.run()
    #face_classif.save("basic")
