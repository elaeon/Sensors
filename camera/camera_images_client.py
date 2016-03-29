import io
import socket
import struct
from PIL import Image
import time
import dlib
from skimage import io as sio

import argparse

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

def get_faces():
    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()
    images = []
    for image in read(num_images=20):
        dets, scores, idx = detector.run(image, 1)
        win.clear_overlay()
        win.set_image(image)
        win.add_overlay(dets)
        if len(dets) > 0:
            for i, d in enumerate(dets):
                images.append((scores[i], image, d, idx[i]))

    images.sort(reverse=True)
    return images

def process_face(url, number_id):
    from face_training import ProcessImages

    images = get_faces()
    if len(images) > 0:
        p = ProcessImages(image_size=90)
        p.save_images(url, number_id, images)

def detect_face():
    from face_training import SVCFace, TensorFace, ProcessImages
    from collections import Counter

    images = get_faces()
    if len(images) > 0:
        face_classif = SVCFace("test_5", image_size=90)
        #face_classif = TensorFace("test_5", image_size=90)
        p = ProcessImages(image_size=90)
        counter = Counter(face_classif.predict_set(p.process_images(images)))
        print(max(counter.items(), key=lambda x: x[1]))

def detect_face_set(dataset_name):
    import os
    import numpy as np
    from face_training import SVCFace, TensorFace, Tensor2LFace, ConvTensorFace, FACE_TEST_FOLDER_PATH

    images = os.listdir(FACE_TEST_FOLDER_PATH)
    classifs = [
        #SVCFace(dataset_name, image_size=90),
        TensorFace(dataset_name, image_size=90),
        #Tensor2LFace(dataset_name, image_size=90),
        #ConvTensorFace(dataset_name, image_size=90)
    ]
    
    for face_classif in classifs:
        print("*#########", face_classif.__class__.__name__)
        images_data = []
        labels = []
        for image in images:
            image_file = os.path.join(FACE_TEST_FOLDER_PATH, image)
            images_data.append(sio.imread(image_file))
            labels.append(image.split("-")[1])
        
        predictions = face_classif.predict_set(images_data)
        face_classif.accuracy(predictions, np.asarray(labels))

def build_dataset(name, directory):
    from face_training import ProcessImages
    p = ProcessImages(90)
    p.load_images(directory)
    p.save_dataset(name)

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--empleado", help="numero de empleado", type=int)
    parser.add_argument("--foto", help="numero de empleado", action="store_true")
    parser.add_argument("--set", help="predice los datos con el dataset como base de conocimiento", type=str)
    parser.add_argument("--build-dataset", help="crea el dataset", type=str)
    args = parser.parse_args()
    if args.empleado:
        process_face("/home/sc/Pictures/face/", args.empleado)
    elif args.foto:
        detect_face()
    elif args.set:
        detect_face_set(args.set)
    elif args.build_dataset:
        build_dataset(args.build_dataset, "/home/sc/Pictures/face/")
