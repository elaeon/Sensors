def check_network():
    import socket
    sock = socket.socket()
    try:
        sock.connect(("www.google.com", 80))
    except socket.error:
        return False
    else:
        return True
    finally:
        sock.close()

def check_carbon(carbon_server, carbon_port):
    import socket
    sock = socket.socket()
    try:
        sock.connect((carbon_server, carbon_port))
    except socket.error:
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
