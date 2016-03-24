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

    images = get_faces()
    if len(images) > 0:
        #face_classif = SVCFace(model="basic")
        face_classif = TensorFace("basic_raw", 10, image_size=90)
        p = ProcessImages(image_size=90)
        image_data = list(p.process_images(images))[0]
        print(face_classif.predict_set(image_data))

def detect_face_set():
    import os
    from face_training import SVCFace, TensorFace, Tensor2LFace, ConvTensorFace, FACE_TEST_FOLDER_PATH

    images = os.listdir(FACE_TEST_FOLDER_PATH)
    #face_classif = SVCFace(model_name="basic_4", image_size=90)
    #face_classif = TensorFace("basic_4", 10, image_size=90)
    face_classif = Tensor2LFace("basic_4", 10, image_size=90)
    #face_classif = ConvTensorFace(model_name="conv", load=True)
    images_data = []
    labels = []
    for image in images:
        image_file = os.path.join(FACE_TEST_FOLDER_PATH, image)
        images_data.append(sio.imread(image_file))
        labels.append(image.split("-")[1])
    
    predictions = face_classif.predict_set(images_data)
    correct = 0
    for label, prediction in zip(labels, predictions):
        print(label, prediction)
        if label == prediction:
            correct += 1
    print("Accuracy: {}%".format(correct*100/len(labels)))

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--empleado", help="numero de empleado", type=int)
    parser.add_argument("--foto", help="numero de empleado", action="store_true")
    parser.add_argument("--set", help="numero de empleado", action="store_true")
    args = parser.parse_args()
    if args.empleado:
        process_face("/home/sc/Pictures/face/", args.empleado)
    elif args.foto:
        detect_face()
    elif args.set:
        detect_face_set()
