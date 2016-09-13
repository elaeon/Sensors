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
    local("cp settings.cfg /tmp/settings.cfg")
    put("/tmp/settings.cfg", "/var/sensors/examples/settings.cfg", use_sudo=True)
    run("sudo chown -R pi:pi /var/sensors/")

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
    run("sudo chmod 755 /var/sensors/examples/*.py")

def test_sensors():
    run("python /var/sensors/test_examples.py")

def check_supervisor():
    run("sudo supervisorctl status")

def install_termopar_paso_1():
    install_ds18b20()
    modprobes_enable()
    reboot(wait=5)

def install_termopar_paso_2():
    nombre_sensor()

def create_permissions_loggers():
    run("touch /tmp/puerta.log")
    run("touch /tmp/temperature_low_one.log")
    run("chown pi:pi /tmp/*.log")
    
def install():
    install_git()
    install_supervisor()
    install_sensors()
    install_bibliotecas_humedad()
    supervisor_conf()    
    change_sensors_file_mod()
    config_hostname()
    create_permissions_loggers()
    reboot(wait=5)
