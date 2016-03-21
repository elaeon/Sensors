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

    def train(self):
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
        score = self.train(
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

    def train(self, train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels):
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

    def train(self, train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels):
        from tensor import BasicTensor
        reg = BasicTensor(self.labels_d, self.image_size, "/home/sc/git/sensors/camera/")
        reg = reg.fit(test_dataset, valid_dataset, 10)
        score = reg.score(train_dataset, train_labels, test_labels, valid_labels, 10)
        self.model = reg
        return score

    def predict_set(self, img):
        self.model.fit(self.test_dataset, self.valid_dataset, 1)
        img = img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
        return self.model.predict(img)
        
    def predict(self, images):
        if self.model is not None:
            self.model.fit(self.test_dataset, self.valid_dataset, 1)
            img = list(self.process_images(images))[0]
            img = img.reshape((-1, self.image_size*self.image_size)).astype(np.float32)
            return self.model.predict(img)

if __name__  == '__main__':
    face_classif = TensorFace()
    #face_classif = SVCFace()
    face_classif.run()
    #face_classif.save("basic")
