from docker import DockerClient
from json import load
from os import system
from time import sleep

# Connects to docker service over tcp
client = DockerClient("tcp://192.168.0.102:4040")

# Runs docker container with first configerations
system('sudo docker run -itd --rm -v /home/kalpit/Projectfiles/Mlopstask3:/home akolkarkalpit/mytensorflow:v1 python -c 0 /home/main.py')


while True:
    if not client.containers.list():
        with open('results.json', 'r') as f:
            data = load(f)
            temp = data['result'][-1]
            val_accuracy = temp['val_accuracy'][-1]
            model = temp['model']
        if val_accuracy < 0.8:
            system('sudo docker run -itd --rm -v /home/kalpit/Projectfiles/Mlopstask3:/home akolkarkalpit/mytensorflow:v1 python -c {} /home/main.py'.format(model+1))
        elif val_accuracy > 0.8:
            
            break
    sleep(60)

print('Crongratulations !!')
