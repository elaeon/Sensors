import io
import socket
import struct
from PIL import Image
import time
import dlib
from skimage import io as sio


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

def detect_face():
    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()

    for image in read():
        #dets = detector(image, 1)
        dets, scores, idx = detector.run(image, 1)
        win.clear_overlay()
        win.set_image(image)
        win.add_overlay(dets)
        if len(dets) > 0:
            print("Number of faces detected: {}".format(len(dets)))
            for i, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    i, d.left(), d.top(), d.right(), d.bottom()))
                print("Detection {}, score: {}, face_type:{}".format(
                    d, scores[i], idx[i]))

    dlib.hit_enter_to_continue()

if __name__  == '__main__':
    detect_face()
    #draw()
