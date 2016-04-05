import io
import socket
import struct
from PIL import Image
import time
import dlib
from skimage import io as sio

import argparse
import face_training
import align_image

# Accept a single connection and make a file-like object out of it
def read(num_images=5):
    client_socket = socket.socket()
    client_socket.connect(('192.168.52.101', 8000))
    client_socket.send(str(num_images))
    connection = client_socket.makefile('rb')
    try:
        while True:
            image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
            if not image_len:
                break
            # Construct a stream to hold the image data and read the image
            # data from the connection
            image_stream = io.BytesIO()
            image_stream.write(connection.read(image_len))
            image = Image.open(image_stream)
            print('Image is %dx%d' % image.size)
            print('%s %s' % (image.format, image.mode))
            image.verify()
            print('Image is verified')
            image = sio.imread(image_stream)
            yield image
    finally:
        connection.close()
        client_socket.close()

def draw():
    win = dlib.image_window()
    start = time.time()
    operation = 0
    for im in read():
        win.clear_overlay()
        win.set_image(im)
        if time.time() - start <= 1:
            operation += 1
        else:
            print("{} fps".format(operation))
            operation = 0
            start = time.time()

def get_faces(images, number_id=None):
    from face_training import FACE_ORIGINAL_PATH
    dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
    align = align_image.FaceAlign(dlibFacePredictor)
    if number_id is None:
        save_path = None
    else:
        save_path = (number_id, FACE_ORIGINAL_PATH)
    return align.process(images, save_path=save_path)

def process_face(url, number_id):
    p_images = get_faces(read(num_images=20), number_id)
    p_images.save_images(url, number_id, p_images.process_images())

def detect_face(face_classif):
    from collections import Counter

    p_images = get_faces(read(num_images=20))
    counter = Counter(face_classif.predict_set(p_images.process_images()))
    print(max(counter.items(), key=lambda x: x[1]))

def detect_face_set(face_classif):
    import os
    import numpy as np

    images = os.listdir(face_training.FACE_TEST_FOLDER_PATH)
    images_data = []
    labels = []
    for image in images:
        image_file = os.path.join(face_training.FACE_TEST_FOLDER_PATH, image)
        images_data.append(sio.imread(image_file))
        labels.append(image.split("-")[1])
    
    predictions = face_classif.predict_set(images_data)
    face_classif.accuracy(list(predictions), np.asarray(labels))

def build_dataset(name, directory):
    from face_training import ProcessImages
    p = ProcessImages(90)
    p.load_images(directory)
    p.save_dataset(name)

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--empleado", help="numero de empleado", type=int)
    parser.add_argument("--foto", help="numero de empleado", action="store_true")
    parser.add_argument("--dataset", help="nombre del dataset a utilizar", type=str)
    parser.add_argument("--set", 
        help="predice los datos con el dataset como base de conocimiento", 
        action="store_true")
    parser.add_argument("--build", help="crea el dataset", action="store_true")
    parser.add_argument("--train", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--classif", help="selecciona el clasificador", type=str)
    args = parser.parse_args()
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = "test_5"


    if args.empleado:
        process_face("/home/sc/Pictures/face/", args.empleado)
    elif args.build:
        build_dataset(dataset_name, "/home/sc/Pictures/face/")
    else:
        classifs = {
            "svc": face_training.SVCFace,
            "tensor": face_training.TensorFace,
            "tensor2": face_training.Tensor2LFace,
            "cnn": face_training.ConvTensorFace
        }
        image_size = 90
        class_ = classifs[args.classif]
        face_classif = class_(dataset_name, image_size=image_size)
        face_classif.batch_size = 10
        print("#########", face_classif.__class__.__name__)
        if args.foto:                  
            detect_face(face_classif)
        elif args.set:
            detect_face_set(face_classif)
        elif args.train:
            face_classif.fit()
            face_classif.train(num_steps=3001)
