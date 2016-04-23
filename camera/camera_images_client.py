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
    client_socket.connect(('192.168.52.102', 8000))
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


def get_faces(images, number_id=None, image_align=True):
    dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
    if image_align is True:
        align = align_image.FaceAlign(dlibFacePredictor)
    else:
        align = align_image.DetectorDlib(dlibFacePredictor)
    if number_id is None:
        save_path = None
    else:
        save_path = (number_id, face_training.FACE_ORIGINAL_PATH)
    return align.process(images, save_path=save_path)


def process_face(url, number_id):
    d = face_training.DataSetBuilder()
    p_images = get_faces(read(num_images=20), number_id)
    images, _ = d.build_train_test((number_id, p_images.process_images(gray=True, blur=True)), sample=False)
    p_images.save_images(url, number_id, images.values())


def detect_face(face_classif):
    from collections import Counter

    p_images = get_faces(read(num_images=20), image_align=True)
    counter = Counter(face_classif.predict_set(p_images.process_images(gray=True, blur=True)))
    if len(counter) > 0:
        print(max(counter.items(), key=lambda x: x[1]))


if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--empleado", help="numero de empleado", type=int)
    parser.add_argument("--foto", help="numero de empleado", action="store_true")
    parser.add_argument("--dataset", help="nombre del dataset a utilizar", type=str)
    parser.add_argument("--test", 
        help="predice los datos test con el dataset como base de conocimiento", 
        action="store_true")
    parser.add_argument("--build", help="crea el dataset", action="store_true")
    parser.add_argument("--rebuild", help="construye el dataset desde las images origen", action="store_true")
    parser.add_argument("--train", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--classif", help="selecciona el clasificador", type=str)
    args = parser.parse_args()
    image_size = 90
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = "test_5"

    if args.empleado:
        process_face("/home/sc/Pictures/face/", args.empleado)
    elif args.build:
        d = face_training.DataSetBuilder()
        d.build_dataset(dataset_name, "/home/sc/Pictures/face/", image_size)
    elif args.rebuild:
        d = face_training.DataSetBuilder()
        d.rebuild_dataset("/home/sc/Pictures/face_o/", image_size, image_align=True)
        d.build_dataset(dataset_name, "/home/sc/Pictures/face/", image_size, channels=None)
    else:        
        classifs = {
            "svc": {
                "name": face_training.SVCFace,
                "params": {"image_size": image_size}},
            "tensor": {
                "name": face_training.TensorFace,
                "params": {"image_size": image_size}},
            "tensor2": {
                "name": face_training.TfLTensor,#face_training.Tensor2LFace,
                "params": {"image_size": image_size}},
            "cnn": {
                "name": face_training.ConvTensor,#ConvTensorFace
                "params": {"num_channels": 1, "image_size": image_size}},
            "residual": {
                "name": face_training.ResidualTensor,
                "params": {"num_channels": 1, "image_size": image_size}}
        }
        class_ = classifs[args.classif]["name"]
        params = classifs[args.classif]["params"]
        face_classif = class_(dataset_name, **params)
        face_classif.batch_size = 10
        print("#########", face_classif.__class__.__name__)
        if args.foto:                  
            detect_face(face_classif)
        elif args.test:
            detect_face_test(face_classif)
        elif args.train:
            face_classif.fit()
            face_classif.train(num_steps=50)
