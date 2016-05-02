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

import align_image

class ProcessImage(object):
    def __init__(self, image, commands):
        self.image = image
        self.pipeline(commands)

    def resize(self, image_size):
        if (self.image_size, self.image_size) < image.shape or\
                    image.shape < (self.image_size, self.image_size):
            self.image = transform.resize(self.image, (image_size, image_size))
        
    def rdb2gray(self):
        self.image = color.rgb2gray(self.image)

    def blur(self, level):
        self.image = filters.gaussian(self.image, level)

    def align_face(self):
        dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
        align = align_image.FaceAlign(dlibFacePredictor)
        self.image = align.process_img(self.image)

    def detector(self):
        dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
        align = align_image.DetectorDlib(dlibFacePredictor)
        self.image = align.process_img(self.image)

    def pipeline(commands):
        for command, values in commands:
            if value is not None:
                getattr(self, command)(value)


class DataSetBuilder(object):
    def __init__(self, name, image_size, channels=None, dataset_path=DATASET_PATH, 
                test_folder_path=FACE_TEST_FOLDER_PATH, 
                train_folder_path=FACE_FOLDER_PATH):
        self.image_size = image_size
        self.images = []
        self.dataset = None
        self.labels = None
        self.channels = channels
        self.test_folder_path = test_folder_path
        self.train_folder_path = train_folder_path
        self.dataset_path = dataset_path
        self.name = name

    def add_img(self, img):
        self.images.append(img)
        #self.dataset.append(img)

    def images_from_directories(self, folder_base):
        images = []
        for directory in os.listdir(folder_base):
            files = os.path.join(folder_base, directory)
            if os.path.isdir(files):
                number_id = directory
                for image_file in os.listdir(files):
                    images.append((number_id, os.path.join(files, image_file)))
        return images

    def images_to_dataset(self, folder_base):
        images = self.images_from_directories(folder_base)
        max_num_images = len(images)
        if self.channels is None:
            self.dataset = np.ndarray(
                shape=(max_num_images, self.image_size, self.image_size), dtype=np.float32)
            dim = (self.image_size, self.image_size)
        else:
            self.dataset = np.ndarray(
                shape=(max_num_images, self.image_size, self.image_size, self.channels), dtype=np.float32)
            dim = (self.image_size, self.image_size, self.channels)
        self.labels = []
        for image_index, (number_id, image_file) in enumerate(images):
            image_data = sio.imread(image_file)
            if image_data.shape != dim:
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            image_data = image_data.astype(float)
            self.dataset[image_index] = preprocessing.scale(image_data)#image_data
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

    def cross_validators(self, dataset, labels, train_size=0.7, valid_size=0.1):
        from sklearn import cross_validation
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            dataset, labels, train_size=train_size, random_state=0)

        valid_size_index = int(round(X_train.shape[0] * valid_size))
        X_validation = X_train[:valid_size_index]
        y_validation = y_train[:valid_size_index]
        X_train = X_train[valid_size_index:]
        y_train = y_train[valid_size_index:]
        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def save_dataset(self, valid_size=.1, train_size=.7):
        train_dataset, valid_dataset, test_dataset, train_labels, valid_labels, test_labels = self.cross_validators(self.dataset, self.labels, train_size=train_size, valid_size=valid_size)
        try:
            f = open(self.dataset_path+self.name, 'wb')
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
            print('Unable to save data to: ', self.dataset_path+self.name, e)
            raise

        print("Test set: {}, Valid set: {}, Training set: {}".format(
            len(test_labels), len(valid_labels), len(train_labels)))

    @classmethod
    def load_dataset(self, name, dataset_path=DATASET_PATH):
        with open(dataset_path+name, 'rb') as f:
            save = pickle.load(f)
            print('Training set DS[{}], labels[{}]'.format(save['train_dataset'].shape, len(save['train_labels'])))
            print('Validation set DS[{}], labels[{}]'.format(save['valid_dataset'].shape, len(save['valid_labels'])))
            print('Test set DS[{}], labels[{}]'.format(save['test_dataset'].shape, len(save['test_labels'])))
            return save

    #def process_images(self, gray=True, blur=True):
    #    for image in self.images:
    #        try:
    #            if gray is True:
    #                image = color.rgb2gray(image)

    #            if (self.image_size, self.image_size) < image.shape or\
    #                image.shape < (self.image_size, self.image_size):
    #                image = transform.resize(image, (self.image_size, self.image_size))
    #                print("Resized")
    #            if blur is True:
    #                image = filters.gaussian(image, .5)
    #            yield image
    #        except ValueError:
    #            pass

    def merge_offset(self, image):
        import random
        bg = np.ones((self.image_size, self.image_size))
        offset = (int(round(abs(bg.shape[0] - image.shape[0]) / 2)), 
                int(round(abs(bg.shape[1] - image.shape[1]) / 2)))
        pos_v, pos_h = offset
        v_range1 = slice(max(0, pos_v), max(min(pos_v + image.shape[0], bg.shape[0]), 0))
        h_range1 = slice(max(0, pos_h), max(min(pos_h + image.shape[1], bg.shape[1]), 0))
        v_range2 = slice(max(0, -pos_v), min(-pos_v + bg.shape[0], image.shape[0]))
        h_range2 = slice(max(0, -pos_h), min(-pos_h + bg.shape[1], image.shape[1]))
        bg2 = bg - 1 + np.average(image) + random.uniform(-np.var(image), np.var(image))
        #print(np.std(image))
        #print(np.var(image))
        bg2[v_range1, h_range1] = bg[v_range1, h_range1] - 1
        bg2[v_range1, h_range1] = bg2[v_range1, h_range1] + image[v_range2, h_range2]
        return bg2

    @classmethod
    def save_images(self, url, number_id, images):
        if not os.path.exists(url):
            os.makedirs(url)
        n_url = "{}{}/".format(url, number_id)
        if not os.path.exists(n_url):
             os.makedirs(n_url)
        for i, image in enumerate(images):
            sio.imsave("{}face-{}-{}.png".format(n_url, number_id, i), image)

    def labels_images(self, url):
        images_data = []
        labels = []
        for number_id, path in self.images_from_directories(url):
            images_data.append(sio.imread(path))
            labels.append(number_id)

        return images_data, labels

    def detector_test(self, face_classif):
        images_data = []
        labels = []
        for number_id, path in self.images_from_directories(self.test_folder_path):
            images_data.append(sio.imread(path))
            labels.append(number_id)
        
        predictions = face_classif.predict_set(images_data)
        face_classif.accuracy(list(predictions), np.asarray(labels))

    def build_dataset(self, from_directory):
        self.images_to_dataset(from_directory)
        self.save_dataset()

    def original_to_images_set(self, url, commands):
        images_data, labels = self.labels_images(url)
        images = (face_training.ProcessImage(image, commands).image for img in images_data)
        image_train, image_test = self.build_train_test(zip(labels, images))

        for number_id, images in image_train.items():
            self.save_images(self.train_folder_path, number_id, images)

        for number_id, images in image_test.items():
            self.save_images(self.test_folder_path, number_id, images)

    def build_train_test(self, process_images, sample=True):
        import random
        images = {}
        images_index = {}
        
        try:
            for number_id, image_array in process_images:
                images.setdefault(number_id, [])
                images[number_id].append(image_array)
        except TypeError:
            #if no faces are detected
            return {}, {}

        if sample is True:
            sample_data = {}
            images_good = {}
            for number_id in images:
                base_indexes = set(range(len(images[number_id])))
                if len(base_indexes) > 3:
                    sample_indexes = set(random.sample(base_indexes, 3))
                else:
                    sample_indexes = set([])
                sample_data[number_id] = [images[number_id][index] for index in sample_indexes]
                images_index[number_id] = base_indexes.difference(sample_indexes)

            for number_id, indexes in images_index.items():
                images_good.setdefault(number_id, [])
                for index in indexes:
                    images_good[number_id].append(images[number_id][index])
            
            return images_good, sample_data
        else:
            return images, {}

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
    def __init__(self, model_name, dataset, image_size=90):
        self.image_size = image_size
        self.model_name = model_name
        self.model = None
        self.le = preprocessing.LabelEncoder()
        self.load_dataset(dataset)

    def detector_test_dataset(self):
        predictions = self.predict_set(self.test_dataset)
        self.accuracy(list(predictions), np.asarray([self.convert_label(label) for label in self.test_labels]))

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

    def load_dataset(self, dataset):
        self.train_dataset = dataset['train_dataset']
        self.train_labels = dataset['train_labels']
        self.valid_dataset = dataset['valid_dataset']
        self.valid_labels = dataset['valid_labels']
        self.test_dataset = dataset['test_dataset']
        self.test_labels = dataset['test_labels']
        self.reformat_all()

class SVCFace(BasicFaceClassif):
    def __init__(self, model_name, dataset, image_size=90, check_point_path=CHECK_POINT_PATH):
        super(SVCFace, self).__init__(model_name, dataset, image_size=image_size)
        self.check_point_path = check_point_path

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
        return self.predict(imgs)

    def transform_img(self, img):
        return img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)

    def predict(self, imgs):
        if self.model is None:
            self.load_model()
        for img in imgs:
            img = self.transform_img(img)
            yield self.convert_label(self.model.predict(img)[0])

    def save_model(self):
        from sklearn.externals import joblib
        joblib.dump(self.model, '{}.pkl'.format(self.check_point_path+self.model_name)) 

    def load_model(self):
        from sklearn.externals import joblib
        self.model = joblib.load('{}.pkl'.format(self.check_point_path+self.model_name))


class BasicTensor(BasicFaceClassif):
    def __init__(self, model_name, dataset, batch_size=None, 
                image_size=90, check_point_path=CHECK_POINT_PATH):
        super(BasicTensor, self).__init__(model_name, dataset, image_size=image_size)
        self.batch_size = batch_size
        self.check_point = check_point_path + self.__class__.__name__ + "/"

    def reformat(self, dataset, labels):
        dataset = self.transform_img(dataset)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels_m = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels_m

    def transform_img(self, img):
        return img.reshape((-1, self.image_size * self.image_size)).astype(np.float32)

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

    def position_index(self, label):
        return np.argmax(label)#[0]

    def predict_set(self, imgs):
        self.batch_size = 1
        self.fit(dropout=False)
        return self.predict(imgs)

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
                yield self.convert_label(classification)

class TfLTensor(TensorFace):
    def fit(self, dropout=False):
        import tflearn
        input_layer = tflearn.input_data(shape=[None, self.image_size*self.image_size])
        dense1 = tflearn.fully_connected(input_layer, 124, activation='tanh',
                                         regularizer='L2', weight_decay=0.001)
        dropout1 = tflearn.dropout(dense1, 0.5)
        dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                        regularizer='L2', weight_decay=0.001)
        dropout2 = tflearn.dropout(dense2, 0.5)
        softmax = tflearn.fully_connected(dropout2, self.num_labels, activation='softmax')

        sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
        #top_k = tflearn.metrics.Top_k(5)
        acc = tflearn.metrics.Accuracy()
        self.net = tflearn.regression(softmax, optimizer=sgd, metric=acc,
                                 loss='categorical_crossentropy')

    def train(self, num_steps=1000):
        import tflearn
        self.model = tflearn.DNN(self.net, tensorboard_verbose=3)
        self.model.fit(self.train_dataset, 
            self.train_labels, 
            n_epoch=num_steps, 
            validation_set=(self.test_dataset, self.test_labels),
            show_metric=True, 
            run_id="dense_model")
        self.save_model()
    
    def save_model(self):
        if not os.path.exists(self.check_point):
            os.makedirs(self.check_point)
        if not os.path.exists(self.check_point + self.model_name + "/"):
            os.makedirs(self.check_point + self.model_name + "/")

        self.model.save('{}{}.ckpt'.format(self.check_point + self.model_name + "/", self.model_name))

    def load_model(self):
        import tflearn
        self.fit()
        self.model = tflearn.DNN(self.net, tensorboard_verbose=3)
        self.model.load('{}{}.ckpt'.format(self.check_point + self.model_name + "/", self.model_name))

    def predict_set(self, imgs):
        return self.predict(imgs)

    def predict(self, imgs):
        if self.model is None:
            self.load_model()
        for img in imgs:
            img = self.transform_img(img)
            yield self.convert_label(self.model.predict(img))

class ConvTensor(TfLTensor):
    def __init__(self, *args, **kwargs):
        self.num_channels = kwargs.get("num_channels", 1)
        self.patch_size = 3
        self.depth = 32
        if "num_channels" in kwargs:
            del kwargs["num_channels"]
        super(ConvTensor, self).__init__(*args, **kwargs)

    def transform_img(self, img):
        return img.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)

    def fit(self, dropout=False):
        import tflearn
        network = tflearn.input_data(
            shape=[None, self.image_size, self.image_size, self.num_channels], name='input')
        network = tflearn.conv_2d(network, self.depth, self.patch_size, activation='relu', regularizer="L2")
        network = tflearn.max_pool_2d(network, 2)
        network = tflearn.local_response_normalization(network)
        network = tflearn.conv_2d(network, self.depth*2, self.patch_size, activation='relu', regularizer="L2")
        network = tflearn.max_pool_2d(network, 2)
        network = tflearn.local_response_normalization(network)
        network = tflearn.fully_connected(network, self.depth*4, activation='tanh')
        network = tflearn.dropout(network, 0.8)
        network = tflearn.fully_connected(network, self.depth*8, activation='tanh')
        network = tflearn.dropout(network, 0.8)
        network = tflearn.fully_connected(network, self.num_labels, activation='softmax')
        #top_k = tflearn.metrics.Top_k(5)
        acc = tflearn.metrics.Accuracy()
        self.net = tflearn.regression(network, optimizer='adam', metric=acc, learning_rate=0.01,
                                 loss='categorical_crossentropy', name='target')

    def train(self, num_steps=1000):
        import tflearn
        self.model = tflearn.DNN(self.net, tensorboard_verbose=3)
        self.model.fit(self.train_dataset, 
            self.train_labels, 
            n_epoch=num_steps, 
            validation_set=(self.test_dataset, self.test_labels),
            show_metric=True, 
            snapshot_step=100,
            run_id="conv_model")
        self.save_model()

class ResidualTensor(TfLTensor):
    def __init__(self, *args, **kwargs):
        self.num_channels = kwargs.get("num_channels", 1)
        self.patch_size = 3
        self.depth = 32
        super(ResidualTensor, self).__init__(*args, **kwargs)

    def transform_img(self, img):
        return img.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)

    def reformat_all(self):
        import tflearn.data_utils as du
        all_ds = np.concatenate((self.train_labels, self.valid_labels, self.test_labels), axis=0)
        self.labels_encode(all_ds)
        self.train_dataset, self.train_labels = self.reformat(
            self.train_dataset, self.le.transform(self.train_labels))
        self.valid_dataset, self.valid_labels = self.reformat(
            self.valid_dataset, self.le.transform(self.valid_labels))
        self.test_dataset, self.test_labels = self.reformat(
            self.test_dataset, self.le.transform(self.test_labels))

        self.train_dataset, mean = du.featurewise_zero_center(self.train_dataset)
        self.test_dataset = du.featurewise_zero_center(self.test_dataset, mean)

        print('RF-Training set', self.train_dataset.shape, self.train_labels.shape)
        print('RF-Validation set', self.valid_dataset.shape, self.valid_labels.shape)
        print('RF-Test set', self.test_dataset.shape, self.test_labels.shape)

    def fit(self, dropout=False):
        import tflearn

        net = tflearn.input_data(shape=[None, self.image_size, self.image_size, self.num_channels])
        net = tflearn.conv_2d(net, self.depth, self.patch_size, activation='relu', bias=False)
        net = tflearn.batch_normalization(net)
        # Residual blocks
        net = tflearn.deep_residual_block(net, self.patch_size, self.depth*2)
        net = tflearn.deep_residual_block(net, 1, self.depth*4, downsample=True)
        net = tflearn.deep_residual_block(net, self.patch_size, self.depth*4)
        net = tflearn.deep_residual_block(net, 1, self.depth*8, downsample=True)
        net = tflearn.deep_residual_block(net, self.patch_size, self.depth*8)
        net_shape = net.get_shape().as_list()
        k_size = [1, net_shape[1], net_shape[2], 1]
        net = tflearn.avg_pool_2d(net, k_size, padding='valid', strides=1)
        # Regression
        net = tflearn.fully_connected(net, self.num_labels, activation='softmax')
        sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=300)
        acc = tflearn.metrics.Accuracy()
        self.net = tflearn.regression(net, optimizer=sgd,
                                metric=acc,
                                loss='categorical_crossentropy',
                                learning_rate=0.1)

    def train(self, num_steps=1000):
        import tflearn
        self.model = tflearn.DNN(self.net, 
            checkpoint_path='model_resnet_mnist', 
            tensorboard_verbose=3,
            max_checkpoints=10)

        self.model.fit(self.train_dataset, 
            self.train_labels, 
            n_epoch=num_steps, 
            validation_set=(self.test_dataset, self.test_labels),
            show_metric=True, 
            snapshot_step=100,
            batch_size=self.batch_size,
            run_id="resnet_mnist")
        self.save_model()


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
