from fabric.api import run, local, env, cd, put, sudo, reboot

# una forma en que se puede correr este comando es 
#(raspberry-env) ramiro@homer:~/Documentos/raspbpi/sensors/buider$ fab -f build.py <funcion> -H 192.168.52.114 --port 1013 #puerto del ssh

env.user = "pi"

def host_type():
    run('uname -s')

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

def backup_configtxt():
    with cd('/boot/'):    
        sudo("cp config.txt config.txt.back")

def envio_configtxt():
    local("cp config.txt /tmp/config.txt")
    put("/tmp/config.txt", "/boot/config.txt", use_sudo=True)

def install_sensors():
    with cd('/var/'):
        run("sudo git clone https://github.com/elaeon/sensors.git")

def setup_settings():
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
    print(resultado.split(" ")[0])

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


def paso_1_install():
    install_git()
    install_supervisor()
    install_sensors()
    supervisor_conf()
    change_sensors_file_mod()
    config_hostname()
    create_permissions_loggers()
    reboot(wait=5)

def paso_2_install_termopar():
    install_termopar_paso_1()

def paso_2_install_humedad():
    install_bibliotecas_humedad()

#paso_3___1:   se copia en la carpeta cp examples/settings.example.cfg  examples/settings.cfg  # el settings.cfg debe estar dentro de la carpeta buider, se modifica el settings.cfg y se pone el nombre del sensor que salio del paso anterior
def paso_3_install():
    setup_settings()
    adiciona_wifi_visitas()
    configura_monitor_peque()

#paso 4__1 raspiconnfig se modifica el timezone y la internacionalizacion

### Lo que continua del paso 4 es poner la red de visitas, y configurar el monitor pequeno, ambos son opcionales
 
def adiciona_wifi_visitas():
    pass_wifi = raw_input("Password de wifi visitas: ")

    linea_1 = "echo \'network={\' >> /etc/wpa_supplicant/wpa_supplicant.conf"
    linea_2 = "echo \'    ssid=\"visitas\"\' >> /etc/wpa_supplicant/wpa_supplicant.conf"
    linea_3 = "echo \'    psk=\"{pass_wifi}\"\' >> /etc/wpa_supplicant/wpa_supplicant.conf".format(pass_wifi=pass_wifi)
    linea_4 = "echo \'}\' >> /etc/wpa_supplicant/wpa_supplicant.conf"

    sudo(linea_1)
    sudo(linea_2)
    sudo(linea_3)
    sudo(linea_4)

def configura_monitor_peque():
    backup_configtxt()
    envio_configtxt()

