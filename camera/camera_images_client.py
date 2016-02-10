import io
import socket
import struct
from PIL import Image
import time
import dlib
from skimage import io as sio

client_socket = socket.socket()
client_socket.connect(('192.168.52.101', 8000))
connection = client_socket.makefile('rb')


# Accept a single connection and make a file-like object out of it
def read():
    try:
        while True:
            # Read the length of the image as a 32-bit unsigned int. If the
            # length is zero, quit the loop
            image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
            if not image_len:
                break
            # Construct a stream to hold the image data and read the image
            # data from the connection
            image_stream = io.BytesIO()
            image_stream.write(connection.read(image_len))
            image = Image.open(image_stream)
            image.verify()
            image = sio.imread(image_stream)
            yield image
    finally:
        connection.close()
        client_socket.close()

def draw():
    #import pylab as pl
    #import matplotlib.image as mpl_image

    win = dlib.image_window()

    #img = None
    for im in read():
        #if img is None:
        #    img = pl.imshow(im)
        #else:
        #    img.set_data(im)
        #pl.pause(.01)
        #pl.draw()
        win.clear_overlay()
        win.set_image(im)
    #win.add_overlay(dets)


def detect_face():
    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()

    for image in read():
        dets = detector(image, 1)
        win.clear_overlay()
        win.set_image(image)
        win.add_overlay(dets)
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

        if len(dets) > 0:
            print("Number of faces detected: {}".format(len(dets)))

            dets, scores, idx = detector.run(image, 1)
            for i, d in enumerate(dets):
                print("Detection {}, score: {}, face_type:{}".format(
                    d, scores[i], idx[i]))

if __name__  == '__main__':
    detect_face()
    #draw()
