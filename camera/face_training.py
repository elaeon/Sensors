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
CHECK_POINT_PATH = "/home/sc/data/face_recog/"
FACE_TEST_FOLDER_PATH = "/home/sc/Pictures/test/"
DATASET_PATH = "/home/sc/data/dataset/"

np.random.seed(133)

class ProcessImages(object):
    def __init__(self, image_size):
        self.image_size = image_size

    def load_images(self, folder):
        folder = FACE_FOLDER_PATH
        images = os.listdir(folder)
        max_num_images = len(images)
        self.dataset = np.ndarray(
            shape=(max_num_images, self.image_size, self.image_size), dtype=np.float32)
        self.labels = np.ndarray(shape=(max_num_images), dtype=np.float32)
        #min_max_scaler = preprocessing.MinMaxScaler()
        for image_index, image in enumerate(images):
            image_file = os.path.join(folder, image)
            image_data = sio.imread(image_file)
            number_id = image.split("-")[1]
            if image_data.shape != (self.image_size, self.image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            image_data = image_data.astype(float)
            self.dataset[image_index, :, :] = preprocessing.scale(image_data)
            self.labels[image_index] = number_id
        num_images = image_index
        print 'Full dataset tensor:', self.dataset.shape
        print 'Mean:', np.mean(self.dataset)
        print 'Standard deviation:', np.std(self.dataset)
        print 'Labels:', self.labels.shape

    def randomize(self, dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    def save_dataset(self, name, valid_size_p=.1, train_size_p=.6):
        total_size = self.dataset.shape[0]
        valid_size = total_size * valid_size_p
        train_size = total_size * train_size_p
        v_t_size = valid_size + train_size
        train_dataset, train_labels = self.randomize(self.dataset[:v_t_size], self.labels[:v_t_size])
        valid_dataset = train_dataset[:valid_size,:,:]
        valid_labels = train_labels[:valid_size]
        train_dataset = train_dataset[valid_size:v_t_size,:,:]
        train_labels = train_labels[valid_size:v_t_size]
        test_dataset, test_labels = self.randomize(
            self.dataset[v_t_size:total_size], self.labels[v_t_size:total_size])
        
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
            test_labels.shape[0], valid_labels.shape[0], train_labels.shape[0]))

    @classmethod
    def load_dataset(self, name):
        with open(DATASET_PATH+name, 'rb') as f:
            save = pickle.load(f)
            print('Training set', save['train_dataset'].shape, save['train_labels'].shape)
            print('Validation set', save['valid_dataset'].shape, save['valid_labels'].shape)
            print('Test set', save['test_dataset'].shape, save['test_labels'].shape)
            return save

    #@classmethod
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

    def save_images(self, url, number_id, images):
        if len(images) > 0:
            for i, image in enumerate(self.process_images(images)):
                sio.imsave(url+"face-{}-{}.png".format(number_id, i), image)

class BasicFaceClassif(object):
    def __init__(self, model_name, load_model=False, image_size=90):
        self.image_size = image_size
        self.model_name = model_name
        self.model = None
        self.load_dataset()

        #if load_model is True:
        #    self.load()

    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
        return dataset, labels

    def reformat_all(self):
        self.train_dataset, self.train_labels = self.reformat(self.train_dataset, self.train_labels)
        self.valid_dataset, self.valid_labels = self.reformat(self.valid_dataset, self.valid_labels)
        self.test_dataset, self.test_labels = self.reformat(self.test_dataset, self.test_labels)
        print('Training set', self.train_dataset.shape, self.train_labels.shape)
        print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
        print('Test set', self.test_dataset.shape, self.test_labels.shape)

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

    #def run(self, batch_size=10):
    #    score = self.train(
    #        self.train_dataset, self.train_labels, self.test_dataset, 
    #        self.test_labels, self.valid_dataset, self.valid_labels,
    #        batch_size=batch_size)
    #    print("Score: {}%".format((score * 100)))

    #def predict(self, images):
    #    if self.model is not None:
    #        img = list(self.process_images(images))[0]
            #sio.imsave("/home/sc/Pictures/face-155-X.png", img)
    #        img = img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
    #        return self.model.predict(img)

    #def predict_set(self, img):
    #    img = img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
    #    return self.model.predict(img)

class SVCFace(BasicFaceClassif):
    def __init__(self, model_name, load_model=False, image_size=90):
        super(SVCFace, self).__init__(model_name, load_model=load, image_size=image_size)

    def train(self, train_dataset, train_labels, test_dataset, test_labels, 
            valid_dataset, valid_labels, batch_size=0):
        from sklearn.linear_model import LogisticRegression
        from sklearn import svm

        reg = LogisticRegression(penalty='l2')
        #reg = svm.SVC(kernel='rbf')
        #reg = svm.LinearSVC(C=1.0, max_iter=1000)
        reg = reg.fit(train_dataset, train_labels)

        score = reg.score(test_dataset, test_labels)
        self.model = reg
        return score

    def save_model(self):
        from sklearn.externals import joblib
        joblib.dump(self.model, '{}.pkl'.format(CHECK_POINT_PATH+self.model_name)) 

    def load_model(self):
        from sklearn.externals import joblib
        self.model = joblib.load('{}.pkl'.format(CHECK_POINT_PATH+self.model_name))


class BasicTensor(BasicFaceClassif):
    def __init__(self, model_name, batch_size, load_model=False, image_size=90):
        super(BasicTensor, self).__init__(model_name, load_model=load_model, image_size=image_size)
        self.batch_size = batch_size
        self.check_point = CHECK_POINT_PATH
        
    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels

    def fit(self):
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
            self.logits = tf.matmul(self.tf_train_dataset, weights) + biases
            self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))

            # Optimizer.
            self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(self.logits)
            self.valid_prediction = tf.nn.softmax(
                tf.matmul(self.tf_valid_dataset, weights) + biases)
            self.test_prediction = tf.nn.softmax(tf.matmul(self.tf_test_dataset, weights) + biases)

    def train(self):
        num_steps = 3001
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
                print "Minibatch accuracy: %.1f%%" % self.accuracy(predictions, batch_labels)
                print "Validation accuracy: %.1f%%" % self.accuracy(
                  self.valid_prediction.eval(), self.valid_labels)
            #print "Test accuracy: %.1f%%" % self.accuracy(self.test_prediction.eval(), self.test_labels)
            score_v = self.accuracy(self.test_prediction.eval(), self.test_labels)
            print('Test accuracy: %.1f' % score_v)
            saver.save(session, '{}{}.ckpt'.format(self.check_point, self.model_name), global_step=step)
            return score_v

class TestTensor(BasicTensor):
    def __init__(self, *args, **kwargs):
        self.num_labels = 10
        super(TestTensor, self).__init__(*args, **kwargs)

class TensorFace(BasicTensor):
    def __init__(self, model_name, batch_size, load_model=False, image_size=90):
        self.labels_d = dict(enumerate(["106", "110", "155"]))
        self.labels_i = {v: k for k, v in self.labels_d.items()}
        self.num_labels = len(self.labels_d)
        super(TensorFace, self).__init__(model_name, batch_size, load_model=load_model, image_size=image_size)

    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
        new_labels = np.asarray([self.labels_i[str(int(label))] for label in labels])
        labels_m = (np.arange(len(self.labels_d)) == new_labels[:,None]).astype(np.float32)
        return dataset, labels_m

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

    def predict_set(self, img):
        self.batch_size = 1
        self.fit()
        img = img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
        return self.predict(img)
        
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
            #print(classification)
            return self.convert_label(classification[0])

    #def predict(self, images):
    #    if self.model is not None:
    #        self.model.fit(self.test_dataset, self.valid_dataset, 1)
    #        img = list(self.process_images(images))[0]
    #        img = img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
    #        return self.model.predict(img)

class Tensor2LFace(TensorFace):
    def load(self):
        from tensor import TowLayerTensor
        self.model = TowLayerTensor(self.labels_d, self.image_size, CHECK_POINT_PATH, model_name=self.model_name)

    def train(self, train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels):
        from tensor import TowLayerTensor
        reg = TowLayerTensor(self.labels_d, self.image_size, CHECK_POINT_PATH, model_name=self.model_name)
        batch_size = 10
        reg.fit(test_dataset, valid_dataset, batch_size)
        score = reg.score(train_dataset, train_labels, test_labels, valid_labels, batch_size)
        self.model = reg
        return score

class ConvTensorFace(TensorFace):
    def __init__(self, model_name, load=False, dataset=False, image_size=90):
        self.num_channels = 1
        super(ConvTensorFace, self).__init__(model_name, load=load, dataset=dataset, image_size=image_size)        

    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)
        new_labels = np.asarray([self.labels_i[str(int(label))] for label in labels])
        labels_m = (np.arange(len(self.labels_d)) == new_labels[:,None]).astype(np.float32)
        return dataset, labels_m

    def load(self):
        from tensor import ConvTensor
        self.model = ConvTensor(self.labels_d, self.image_size, CHECK_POINT_PATH, model_name=self.model_name)

    def train(self, train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels):
        from tensor import ConvTensor
        reg = ConvTensor(self.labels_d, self.image_size, CHECK_POINT_PATH, model_name=self.model_name)
        batch_size = 10
        patch_size = 5
        depth = 90
        num_hidden = 64
        reg.fit(test_dataset, valid_dataset, batch_size, patch_size, depth, self.num_channels, num_hidden)
        score = reg.score(train_dataset, train_labels, test_labels, valid_labels, batch_size)
        self.model = reg
        return score

    #def predict_set(self, img):
    #    self.model.fit(self.test_dataset, self.valid_dataset, 1, 5, 90, 1, 64)
    #    img = img.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)
    #    return self.model.predict(img)
        
    #def predict(self, images):
    #    if self.model is not None:
    #        self.model.fit(self.test_dataset, self.valid_dataset, 1)
    #        img = list(self.process_images(images))[0]
    #        img = img.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)
    #        return self.model.predict(img)

if __name__  == '__main__':
    #face_classif = ConvTensorFace(model_name="conv")
    face_classif = TensorFace("basic_raw", 10, image_size=90)
    #face_classif = Tensor2LFace(model_name="layer")
    #face_classif = SVCFace()
    face_classif.fit()
    face_classif.train()
    #face_classif.save_dataset()
    #face_classif.save("basic")
