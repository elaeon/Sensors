def check_network(host, port):
    import socket
    try:
        sock = socket.socket()
        sock.settimeout(10.0)
        sock.connect((host, port))
    except socket.error:
        return False
    except socket.timeout:
        return False
    else:
        return True
    finally:
        sock.close()

def get_settings(base_path, directory=None):
    import ConfigParser
    import os
    config = ConfigParser.ConfigParser()
    if directory is None:
        settings_path = os.path.join(os.path.dirname(os.path.abspath(base_path)), "settings.cfg")
    else:
        settings_path = os.path.join(os.path.dirname(os.path.abspath(base_path)), directory, "settings.cfg")
    config.read(settings_path)
    return config
