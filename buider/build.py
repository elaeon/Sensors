from fabric.api import run, local, env, cd

env.user = "pi"

def config_hostname():
    hostname = raw_input("Hostname: ")
    command = "sudo echo '{hostname}' > /etc/hostname".format(hostname=hostname)
    run(command)

def install_git():   
    run("sudo apt-get install git")

def install_supervisor():
    run("sudo apt-get install supervisor")

def supervisor_conf():
    local("cp sensors.conf /tmp/sensors.conf")
    put("/tmp/sensors.conf", "/etc/supervisor/conf.d/sensors.conf")

def install_sensors():
    with cd('/var/'):
        run("sudo git clone https://github.com/elaeon/sensors.git")


def install():
    install_git()
    install_supervisor()
    install_sensors()
    supervisor_conf()
    config_hostname()
