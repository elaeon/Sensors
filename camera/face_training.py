import os
import numpy as np
from skimage import io as sio
from skimage import color
from skimage import filters
from skimage import transform
from sklearn import preprocessing

import tensorflow as tf
import cPickle as pickle

FACE_FOLDER_PATH = "/home/sc/Pictures/face/"
FACE_ORIGINAL_PATH = "/home/sc/Pictures/face_o/"
CHECK_POINT_PATH = "/home/sc/data/face_recog/"
FACE_TEST_FOLDER_PATH = "/home/sc/Pictures/test/"
DATASET_PATH = "/home/sc/data/dataset/"

np.random.seed(133)

class ProcessImages(object):
    def __init__(self, image_size):
        self.image_size = image_size
        self.images = []

    def add_img(self, img):
        self.images.append(img)

    def images_from_directories(self, folder_base):
        images = []
        for directory in os.listdir(folder_base):
            files = os.path.join(folder_base, directory)
            if os.path.isdir(files):
                number_id = directory
                for image_file in os.listdir(files):
                    images.append((number_id, os.path.join(files, image_file)))
        return images

    def load_images(self, folder_base):
        images = self.images_from_directories(folder_base)
        max_num_images = len(images)
        self.dataset = np.ndarray(
            shape=(max_num_images, self.image_size, self.image_size), dtype=np.float32)
        self.labels = []
        for image_index, (number_id, image_file) in enumerate(images):
            image_data = sio.imread(image_file)
            if image_data.shape != (self.image_size, self.image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            image_data = image_data.astype(float)
            self.dataset[image_index, :, :] = image_data#preprocessing.scale(image_data)
            self.labels.append(number_id)
        print 'Full dataset tensor:', self.dataset.shape
        print 'Mean:', np.mean(self.dataset)
        print 'Standard deviation:', np.std(self.dataset)
        print 'Labels:', len(self.labels)

    def randomize(self, dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    def cross_validators(self, dataset, labels, train_size=0.6, valid_size=0.1):
        from sklearn import cross_validation
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            dataset, labels, train_size=train_size, random_state=0)

        test_size = 1 - train_size - valid_size
        if 0 < test_size < 1:
            test_size_proportion = test_size / (test_size + valid_size)
            test_size_index = X_test.shape[0] * test_size_proportion
        else:
            raise Exception
        X_validation = X_test[test_size_index:,:,:]
        y_validation = y_test[int(test_size_index):]
        X_test = X_test[:test_size_index,:,:]
        y_test = y_test[:int(test_size_index)]        
        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def save_dataset(self, name, valid_size=.1, train_size=.6):
        train_dataset, valid_dataset, test_dataset, train_labels, valid_labels, test_labels = self.cross_validators(self.dataset, self.labels, train_size=train_size, valid_size=valid_size)
        try:
            f = open(DATASET_PATH+name, 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
                }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to: ', DATASET_PATH+name, e)
            raise

        print("Test set: {}, Valid set: {}, Training set: {}".format(
            len(test_labels), len(valid_labels), len(train_labels)))

    @classmethod
    def load_dataset(self, name):
        with open(DATASET_PATH+name, 'rb') as f:
            save = pickle.load(f)
            print('Training set', save['train_dataset'].shape, len(save['train_labels']))
            print('Validation set', save['valid_dataset'].shape, len(save['valid_labels']))
            print('Test set', save['test_dataset'].shape, len(save['test_labels']))
            return save

    def process_images(self):
        for image in self.images:
            try:
                #print(image.shape)
                img_gray = color.rgb2gray(image)
                if (self.image_size, self.image_size) < img_gray.shape or\
                    img_gray.shape < (self.image_size, self.image_size):
                    img_gray = transform.resize(img_gray, (self.image_size, self.image_size))
                    print("Resized")
                img_gray = filters.gaussian(img_gray, .5)
                yield img_gray
            except ValueError:
                pass

    def save_images(self, url, number_id, images):
        if not os.path.exists(url):
            os.makedirs(url)
        n_url = "{}{}/".format(url, number_id)
        if not os.path.exists(n_url):
             os.makedirs(n_url)
        for i, image in enumerate(images):
            sio.imsave("{}face-{}-{}.png".format(n_url, number_id, i), image)

class Measure(object):
    def __init__(self, predictions, labels):
        if len(labels.shape) > 1:
            self.labels = np.argmax(labels, 1)
            self.predictions = np.argmax(predictions, 1)
        else:
            self.labels = labels
            self.predictions = predictions
        self.average = "macro"

    def accuracy(self):
        from sklearn.metrics import accuracy_score
        return accuracy_score(self.labels, self.predictions)

    #false positives
    def precision(self):
        from sklearn.metrics import precision_score
        return precision_score(self.labels, self.predictions, average=self.average, pos_label=None)

    #false negatives
    def recall(self):
        from sklearn.metrics import recall_score
        return recall_score(self.labels, self.predictions, average=self.average, pos_label=None)

    #weighted avg presicion and recall
    def f1(self):
        from sklearn.metrics import f1_score
        return f1_score(self.labels, self.predictions, average=self.average, pos_label=None)

    def print_all(self):
        print("#############")
        print("Accuracy: {}%".format(self.accuracy()*100))
        print("Precision: {}%".format(self.precision()*100))
        print("Recall: {}%".format(self.recall()*100))
        print("F1: {}%".format(self.f1()*100))
        print("#############")

class BasicFaceClassif(object):
    def __init__(self, model_name, image_size=90):
        self.image_size = image_size
        self.model_name = model_name
        self.model = None
        self.le = preprocessing.LabelEncoder()
        self.load_dataset()

    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
        return dataset, labels

    def labels_encode(self, labels):
        self.le.fit(labels)
        self.num_labels = self.le.classes_.shape[0]

    def position_index(self, label):
        return label

    def convert_label(self, label):
        #[0, 0, 1.0] -> 155
        return self.le.inverse_transform(self.position_index(label))

    def reformat_all(self):
        all_ds = np.concatenate((self.train_labels, self.valid_labels, self.test_labels), axis=0)
        self.labels_encode(all_ds)
        self.train_dataset, self.train_labels = self.reformat(
            self.train_dataset, self.le.transform(self.train_labels))
        self.valid_dataset, self.valid_labels = self.reformat(
            self.valid_dataset, self.le.transform(self.valid_labels))
        self.test_dataset, self.test_labels = self.reformat(
            self.test_dataset, self.le.transform(self.test_labels))
        print('RF-Training set', self.train_dataset.shape, self.train_labels.shape)
        print('RF-Validation set', self.valid_dataset.shape, self.valid_labels.shape)
        print('RF-Test set', self.test_dataset.shape, self.test_labels.shape)

    def accuracy(self, predictions, labels):
        measure = Measure(predictions, labels)
        measure.print_all()
        return measure.accuracy()

    def load_dataset(self):
        data = ProcessImages.load_dataset(self.model_name)
        self.train_dataset = data['train_dataset']
        self.train_labels = data['train_labels']
        self.valid_dataset = data['valid_dataset']
        self.valid_labels = data['valid_labels']
        self.test_dataset = data['test_dataset']
        self.test_labels = data['test_labels']
        self.reformat_all()
        del data

class SVCFace(BasicFaceClassif):
    def __init__(self, model_name, image_size=90):
        super(SVCFace, self).__init__(model_name, image_size=image_size)

    def fit(self):
        #from sklearn.linear_model import LogisticRegression
        from sklearn import svm
        #reg = LogisticRegression(penalty='l2')
        reg = svm.LinearSVC(C=1.0, max_iter=1000)
        reg = reg.fit(self.train_dataset, self.train_labels)
        self.model = reg

    def train(self, num_steps=0):
        predictions = self.model.predict(self.test_dataset)
        score = self.accuracy(predictions, self.test_labels)
        print('Test accuracy: %.1f%%' % (score*100))
        self.save_model()
        return score

    def predict_set(self, imgs):
        if self.model is None:
            self.load_model()
        return (self.predict(img) for img in imgs)

    def transform_img(self, img):
        return img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)

    def predict(self, img):
        img = self.transform_img(img)
        if self.model is None:
            self.load_model()
        return self.convert_label(self.model.predict(img)[0])

    def save_model(self):
        from sklearn.externals import joblib
        joblib.dump(self.model, '{}.pkl'.format(CHECK_POINT_PATH+self.model_name)) 

    def load_model(self):
        from sklearn.externals import joblib
        self.model = joblib.load('{}.pkl'.format(CHECK_POINT_PATH+self.model_name))


class BasicTensor(BasicFaceClassif):
    def __init__(self, model_name, batch_size=None, image_size=90):
        super(BasicTensor, self).__init__(model_name, image_size=image_size)
        self.batch_size = batch_size
        self.check_point = CHECK_POINT_PATH + self.__class__.__name__ + "/"

    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels

    def fit(self, dropout=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            self.tf_train_dataset = tf.placeholder(tf.float32,
                                            shape=(self.batch_size, self.image_size * self.image_size))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
            self.tf_valid_dataset = tf.constant(self.valid_dataset)
            self.tf_test_dataset = tf.constant(self.test_dataset)

            # Variables.
            weights = tf.Variable(
                tf.truncated_normal([self.image_size * self.image_size, self.num_labels]))
            biases = tf.Variable(tf.zeros([self.num_labels]))

            # Training computation.
            if dropout is True:
                hidden = tf.nn.dropout(self.tf_train_dataset, 0.5, seed=66478)
                self.logits = tf.matmul(hidden, weights) + biases
            else:
                self.logits = tf.matmul(self.tf_train_dataset, weights) + biases
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))

            regularizers = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
            self.loss += 5e-4 * regularizers

            # Optimizer.
            self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(self.logits)
            self.valid_prediction = tf.nn.softmax(
                tf.matmul(self.tf_valid_dataset, weights) + biases)
            self.test_prediction = tf.nn.softmax(tf.matmul(self.tf_test_dataset, weights) + biases)

    def train(self, num_steps=3001):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()
            print "Initialized"
            for step in xrange(num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)
                # Generate a minibatch.
                batch_data = self.train_dataset[offset:(offset + self.batch_size), :]
                batch_labels = self.train_labels[offset:(offset + self.batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
                _, l, predictions = session.run(
                [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (step % 500 == 0):
                    print "Minibatch loss at step", step, ":", l
                    print "Minibatch accuracy: %.1f%%" % (self.accuracy(predictions, batch_labels)*100)
                    print "Validation accuracy: %.1f%%" % (self.accuracy(
                      self.valid_prediction.eval(), self.valid_labels)*100)
            score_v = self.accuracy(self.test_prediction.eval(), self.test_labels)
            self.save_model(saver, session, step)
            return score_v

    def save_model(self, saver, session, step):
        if not os.path.exists(self.check_point):
            os.makedirs(self.check_point)
        if not os.path.exists(self.check_point + self.model_name + "/"):
            os.makedirs(self.check_point + self.model_name + "/")
        
        saver.save(session, 
                '{}{}.ckpt'.format(self.check_point + self.model_name + "/", self.model_name), 
                global_step=step)

class TensorFace(BasicTensor):
    def __init__(self, *args, **kwargs):
        super(TensorFace, self).__init__(*args, **kwargs)

    def reformat(self, dataset, labels):
        dataset = self.transform_img(dataset)
        labels_m = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels_m

    def position_index(self, label):
        return np.argmax(label, 1)[0]

    def predict_set(self, imgs):
        self.batch_size = 1
        self.fit(dropout=False)
        return self.predict(imgs)
        
    def transform_img(self, img):
        return img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)

    def predict(self, imgs):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.check_point + self.model_name + "/")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("...no checkpoint found...")

            for img in imgs:
                img = self.transform_img(img)
                feed_dict = {self.tf_train_dataset: img}
                classification = session.run(self.train_prediction, feed_dict=feed_dict)
                #print(classification)
                yield self.convert_label(classification)

class Tensor2LFace(TensorFace):
    def __init__(self, *args, **kwargs):
        super(Tensor2LFace, self).__init__(*args, **kwargs)
        self.num_hidden = 1024

    def layers(self, dropout):
        W1 = tf.Variable(
            tf.truncated_normal([self.image_size * self.image_size, self.num_hidden], stddev=1.0 / 10), 
            name='weights')
        b1 = tf.Variable(tf.zeros([self.num_hidden]), name='biases')
        hidden = tf.nn.relu(tf.matmul(self.tf_train_dataset, W1) + b1)

        W2 = tf.Variable(
            tf.truncated_normal([self.num_hidden, self.num_labels]))
        b2 = tf.Variable(tf.zeros([self.num_labels]))

        if dropout is True:
            hidden = tf.nn.dropout(hidden, 0.5, seed=66478)
        self.logits = tf.matmul(hidden, W2) + b2
        return W1, b1, W2, b2

    def fit(self, dropout=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_train_dataset = tf.placeholder(tf.float32,
                                            shape=(self.batch_size, self.image_size * self.image_size))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
            self.tf_valid_dataset = tf.constant(self.valid_dataset)
            self.tf_test_dataset = tf.constant(self.test_dataset)

            W1, b1, W2, b2 = self.layers(dropout)

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


class ConvTensorFace(TensorFace):
    def __init__(self, *args, **kwargs):
        self.num_channels = 1
        self.patch_size = 5
        self.depth = 32
        super(ConvTensorFace, self).__init__(*args, **kwargs)        
        self.num_hidden = 64

    def transform_img(self, img):
        return img.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)

    def layers(self, data, layer1_weights, layer1_biases, layer2_weights, layer2_biases, 
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

    def fit(self, dropout=True):
        import math
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_train_dataset = tf.placeholder(
                tf.float32, shape=(self.batch_size, self.image_size, self.image_size, self.num_channels))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
            self.tf_valid_dataset = tf.constant(self.valid_dataset)
            self.tf_test_dataset = tf.constant(self.test_dataset)

            # Variables.
            layer3_size = int(math.ceil(self.image_size / 4.))
            layer1_weights = tf.Variable(tf.truncated_normal(
                [self.patch_size, self.patch_size, self.num_channels, self.depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([self.depth]))
            layer2_weights = tf.Variable(tf.truncated_normal(
                [layer3_size * layer3_size * self.depth, self.num_hidden], stddev=0.1)) # 4 num of ksize
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden]))
            layer3_weights = tf.Variable(tf.truncated_normal(
                [self.num_hidden, self.num_labels], stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_labels]))

            self.logits = self.layers(self.tf_train_dataset, layer1_weights, 
                layer1_biases, layer2_weights, layer2_biases, layer3_weights, 
                layer3_biases, dropout=dropout)

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
              batch * self.batch_size,  # Current index into the dataset.
              self.train_labels.shape[0],          # train_labels.shape[0] Decay step.
              0.95,                # Decay rate.
              staircase=True)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss,
                global_step=batch)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(self.logits)
            self.valid_prediction = tf.nn.softmax(self.layers(self.tf_valid_dataset, layer1_weights, 
                layer1_biases, layer2_weights, layer2_biases, layer3_weights, 
                layer3_biases))
            self.test_prediction = tf.nn.softmax(self.layers(self.tf_test_dataset, layer1_weights, 
                layer1_biases, layer2_weights, layer2_biases, layer3_weights, 
                layer3_biases))

    def train(self, num_steps=0):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()
            print("Initialized")
            for step in xrange(int(15 * self.train_labels.shape[0]) // self.batch_size):
                offset = (step * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)
                batch_data = self.train_dataset[offset:(offset + self.batch_size), :, :, :]
                batch_labels = self.train_labels[offset:(offset + self.batch_size), :]
                feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
                _, l, predictions = session.run(
                [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (step % 5000 == 0):
                    print "Minibatch loss at step", step, ":", l
                    print "Minibatch accuracy: %.1f%%" % self.accuracy(predictions, batch_labels)
                    print "Validation accuracy: %.1f%%" % self.accuracy(
                    self.valid_prediction.eval(), self.valid_labels)
            score = self.accuracy(self.test_prediction.eval(), self.test_labels)
            #print('Test accuracy: %.1f' % score)
            self.save_model(saver, session, step)
            return score
