from PIL import Image
from socket import *
import numpy as np
import threading
import argparse
import serial
import time
import cv2
import os

classes = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]

objects = {
    'aviao': 'aeroplane',
    'bicicleta': 'bicycle',
    'passaro': 'bird',
    'canoa': 'boat',
    'garrafa': 'bottle',
    'onibus': 'bus',
    'carro': 'car',
    'gato': 'cat',
    'cadeira': 'chair',
    'vaca': 'cow',
    'mesa': 'diningtable',
    'cachorro': 'dog',
    'cavalo': 'horse',
    'motocicleta': 'motorbike', 'moto': 'motorbike',
    'pessoa': 'person', 'humano': 'person',
    'planta': 'pottedplant',
    'ovelha': 'sheep',
    'sofa': 'sofa',
    'trem': 'train',
    'tv': 'tvmonitor', 'televisao': 'tvmonitor'
}

arduino_cmds = {
    'forward': '1',
    'turn_right': '2',
    'turn_left': '3',
    'backward': '4',
    'beep': '5',
    'find': '6'
}

colors = np.random.uniform(0, 255, size=(len(classes), 3))
socket_num = 3000
arduin_com = serial.Serial('/dev/ttyACM0', 9600)
msecs_per_degree = 1
msecs_per_unit = 10
#log = open('report.log', 'w')

def predict(image, net, obj):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if idx >= len(classes):
		continue
	    if classes[idx] == obj:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                coords = list(box.astype('int')) # Xi, Yi, Xf, Yf
                for j in range(0, 4):
                    coords[j] = max(0, min(300, coords[j]))
                return coords
    return False

def capture_predict(obj):
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
    os.system('fswebcam -r 299x299 --jpeg 85 -S 10 -q teste.jpg')
    image = cv2.imread('teste.jpg')
    return predict(image, net, obj)

def arduino_move(action, msecs):
    units = int(round(float(msecs) / msecs_per_unit))
    print arduino_cmds[action], units
    while units > 0:
        arduin_com.write(arduino_cmds[action] + str(min(9, units-1)))
        units -= min(10, units)
    time.sleep((msecs * 1.2) / 1000.0)

def arduino_beep(times):
    arduin_com.write('5' + str(min(9, max(0, times-1))))

def arduino_goto_obj():
    arduin_com.write('6')

# rotate 300ms right per frame (3900ms = ~360dg)
def search(obj):
    arduino_beep(2)
    bounding_box = False
    for i in range(0, 13):
        bounding_box = capture_predict(obj)
        if bounding_box:
            break
        arduino_move('turn_right', 300)
    if bounding_box:
        arduino_beep(3)
    else:
        arduino_beep(1)

def receive(rcv_socket):
    data = rcv_socket.recv(2048)
    cmd = list(str(data.decode('utf-8')).lower())
    cmd = ''.join([c for c in cmd if c != '\x00'])
    words = cmd.split(' ')
    if len(words) > 1 and any(word in words for word in [ 'cade', 'pegue', 'encontre', 'ache', 'procure' ]):
        obj = words[2 if len(words) >= 3 else 1]
        if obj in objects:
            search(objects[obj])
    elif cmd in [ 'va para frente', 'siga em frente', 'va reto', 'va em frente', 'taca-le pau', 'taca-lhe pau', 'tacale pau' ]:
        arduino_move('forward', 3000)
    elif cmd in [ 'vire para direita', 'vire para a direita' ]:
     	arduino_move('turn_right', 1900)
    elif cmd in [ 'vire para esquerda', 'vire para a esquerda' ]:
        arduino_move('turn_left', 1900)
    elif cmd in [ 'va para tras', 'volte', 'recue', 'volte atras' ]:
        print 'backward'
        arduino_move('backward', 2500)
    elif cmd in [ 'faca barulho', 'apite', 'alerte', 'grite', 'seja insuportavel' ]:
        arduino_beep(3)
    elif cmd in [ 'mostre todas as funcionalidades', 'se apresente', 'se exiba', 'mostre tudo o que sabe fazer', 'mostre quem e o bonzao', 'ulte', 'solte o ultimate', 'aperta o r', 'aperta o q', 'aperta esse errre', 'aperta esse r' ]:
        arduino_move('forward', 3000)
        arduino_move('turn_left', 1900)
        arduino_move('turn_right', 1900)
        arduino_move('backward', 2500)
        arduino_beep(3)
    elif cmd in [ 'corre berg', 'ande ate nao dar mais', 'va para frente ate ser bloqueado', 'va pra frente ate ser bloqueado', 'ao infinito e alem' ]:
        arduino_goto_obj()
    elif cmd in [ 'desativar', 'desligar' ]:
        arduino_beep(1)
	#log.close()
        time.sleep(1)
        os.system('sudo shutdown -h 0')
    rcv_socket.send('ack'.encode(encoding='utf-8', errors='ignore'))
    rcv_socket.close()

'''
for i in range(0, 13):
    #os.system('fswebcam -r 299x299 --jpeg 85 -S 10 -q img' + str(i) + '.jpg')
    #arduino_move('turn_right', 300)
'''

my_socket = socket(AF_INET,SOCK_STREAM)
my_socket.bind(('',socket_num))
my_socket.listen(1)

time.sleep(2)
arduino_beep(2)

while True:
    rcv_socket, addr = my_socket.accept()
    new_thread = threading.Thread(target=receive, args=(rcv_socket,))
    new_thread.start()
my_socket.close()
