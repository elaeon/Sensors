import io
import socket
import struct
import time
import picamera

while True:
    try:
        with picamera.PiCamera() as camera:
            camera.resolution = (640, 480)
            camera.framerate = 30
            camera.rotation = 180

            server_socket = socket.socket()
            server_socket.bind(('0.0.0.0', 8000))
            server_socket.listen(0)

            connection = server_socket.accept()[0].makefile('wb')

            print("Starting warming")
            time.sleep(1)

            stream = io.BytesIO()
            for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                # Write the length of the capture to the stream and flush to
                # ensure it actually gets sent
                connection.write(struct.pack('<L', stream.tell()))
                connection.flush()
                # Rewind the stream and send the image data over the wire
                stream.seek(0)
                connection.write(stream.read())
                # Reset the stream for the next capture
                stream.seek(0)
                stream.truncate()
        # Write a length of zero to the stream to signal we're done
        connection.write(struct.pack('<L', 0))
        connection.close()
    except socket.error:
        print("Close conection")
        time.sleep(1)
    finally:
        #connection.close()
        server_socket.close()
