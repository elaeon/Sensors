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
        labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
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

    def train(self, train_dataset, train_labels, test_dataset, test_labels):
        pass

    def run(self):
        dataset, labels = self.load_images()
        total_size = dataset.shape[0]
        valid_size = total_size * .1
        train_size = total_size * .6
        v_t_size = valid_size+train_size
        train_dataset, train_labels = self.randomize(dataset[:v_t_size], labels[:v_t_size])
        valid_dataset = train_dataset[:valid_size,:,:]
        valid_labels = train_labels[:valid_size]
        train_dataset = train_dataset[valid_size:v_t_size,:,:]
        train_labels = train_labels[valid_size:v_t_size]
        test_dataset, test_labels = self.randomize(dataset[v_t_size:total_size], labels[v_t_size:total_size])
        score = self.train(train_dataset, train_labels, test_dataset, test_labels)
        test_size = test_labels.shape[0]
        print("Test set: {}, Valid set: {}, Training set: {}".format(test_size, valid_size, train_size))
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
    def train(self, train_dataset, train_labels, test_dataset, test_labels):
        #from sklearn.linear_model import LogisticRegression
        from sklearn import svm
        train = train_dataset.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
        #reg = LogisticRegression()
        #reg = svm.SVC(kernel='rbf')
        reg = svm.LinearSVC(C=1.0, max_iter=1000)
        reg = reg.fit(train, train_labels)

        test = test_dataset.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
        score = reg.score(test, test_labels)
        self.model = reg
        return score

class TensorFace(BasicFaceClassif):
    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(labels.shape[0]) == labels[:,None]).astype(np.float32)
        return dataset, labels

    def train(self, train_dataset, train_labels, test_dataset, test_labels):
        train, train_labels = self.reformat(train_dataset, train_labels)
        #valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
        print(test_labels)
        test, test_labels = self.reformat(test_dataset, test_labels)
        print(test_labels)
        #reg = svm.LinearSVC(C=1.0, max_iter=1000)
        #reg = reg.fit(train, train_labels)
        #score = reg.score(test, test_labels)
        #self.model = reg
        #self.fit(train_dataset, train_labels, valid_dataset, test_dataset)
        #return score
        #train_dataset, train_labels = reformat(train_dataset, train_labels)
        #valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
        #test_dataset, test_labels = reformat(test_dataset, test_labels)
        #print 'Training set', train_dataset.shape, train_labels.shape
        #print 'Validation set', valid_dataset.shape, valid_labels.shape
        #print 'Test set', test_dataset.shape, test_labels.shape

    def fit(self, train_dataset, train_labels, valid_dataset, test_dataset, test_labels):
        import tensorflow as tf
        # With gradient descent training, even this much data is prohibitive.
        # Subset the training data for faster turnaround.
        train_subset = 10000
        num_labels = train_labels.shape[0]
        graph = tf.Graph()
        with graph.as_default():
            # Input data.
            # Load the training, validation and test data into constants that are
            # attached to the graph.
            tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
            tf_train_labels = tf.constant(train_labels[:train_subset])
            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)

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

        num_steps = 801
        with tf.Session(graph=graph) as session:
            # This is a one-time operation which ensures the parameters get initialized as
            # we described in the graph: random weights for the matrix, zeros for the
            # biases. 
            tf.initialize_all_variables().run()
            print 'Initialized'
            for step in xrange(num_steps):
                # Run the computations. We tell .run() that we want to run the optimizer,
                # and get the loss value and the training predictions returned as numpy
                # arrays.
                _, l, predictions = session.run([optimizer, loss, train_prediction])
                if (step % 100 == 0):
                    print 'Loss at step', step, ':', l
                    print 'Training accuracy: %.1f%%' % self.accuracy(
                    predictions, train_labels[:train_subset, :])
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                    print 'Validation accuracy: %.1f%%' % self.accuracy(
                    valid_prediction.eval(), valid_labels)
            print 'Test accuracy: %.1f%%' % self.accuracy(test_prediction.eval(), test_labels)

    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])

if __name__  == '__main__':
    face_classif = TensorFace()#SVCFace()#BasicFaceClassif()
    face_classif.run()
    #face_classif.save("basic")
