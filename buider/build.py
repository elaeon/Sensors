from fabric.api import run, local, env, cd, put, sudo, reboot

env.user = "pi"

def config_hostname():
    hostname = raw_input("Hostname: ")
    command = "echo '{hostname}' > /etc/hostname".format(hostname=hostname)
    sudo(command)

    command_1 = "echo '127.0.0.1\t{hostname}' >> /etc/hosts".format(hostname=hostname)    
    sudo(command_1)

def install_git():   
    run("sudo apt-get install git")

def install_supervisor():
    run("sudo apt-get install supervisor")

def supervisor_conf():
    local("cp sensors.conf /tmp/sensors.conf")
    put("/tmp/sensors.conf", "/etc/supervisor/conf.d/sensors.conf", use_sudo=True)

def install_sensors():
    with cd('/var/'):
        run("sudo git clone https://github.com/elaeon/sensors.git")

def install_ds18b20():
    command = "echo 'dtoverlay=w1-gpio' >> /boot/config.txt"
    sudo(command)

def modprobes_enable():
    command_1 = "modprobe w1-gpio"
    command_2 = "modprobe w1-therm"
    sudo(command_1)
    sudo(command_2)

def nombre_sensor():
    command_1 = "ls /sys/bus/w1/devices"
    resultado = run(command_1)
    print resultado.split(" ")[0]

def install_bibliotecas_humedad():
    sudo("apt-get install build-essential python-dev")
    with cd('/home/pi/'):
        run("git clone https://github.com/adafruit/Adafruit_Python_DHT")
    with cd('/home/pi/Adafruit_Python_DHT'):
        sudo("python setup.py install")

def change_sensors_file_mod():
    run("sudo chmod 755 /var/sensors/examples/")

def test():
    run("python /var/sensors/test_examples.py")

def check_scripts():
    run("supervisotctl status")

def install():
    install_git()
    install_supervisor()
    install_sensors()
    install_ds18b20()
    modprobes_enable()
    install_bibliotecas_humedad()
    supervisor_conf()    
    change_sensors_file_mod()
    config_hostname()
    reboot(wait=5)
