import numpy as np
import argparse
import cv2
import serial
import threading
import os
import time
from socket import *
from PIL import Image

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

objects = {  "aviao": "aeroplane",
                    "bicicleta": "bicycle",
                    "passaro": "bird",
                    "canoa": "boat",
                    "garrafa": "bottle",
                    "onibus": "bus",
                    "carro": "car",
                    "gato": "cat",
                    "cadeira": "chair",
                    "vaca": "cow",
                    "mesa": "diningtable",
                    "cachorro": "dog",
                    "cavalo": "horse",
                    "motocicleta": "motorbike", "moto": "motorbike",
                    "pessoa": "person", "humano": "person",
                    "planta": "pottedplant",
                    "ovelha": "sheep",
                    "sofa": "sofa",
                    "trem": "train",
                    "tv": "tvmonitor", "televisao": "tvmonitor" }

colors = np.random.uniform(0, 255, size=(len(classes), 3))
socket_num = 3001
arduin_com = serial.Serial('/dev/ttyACM0', 9600)
msecs_per_degree = 1
msecs_per_unit = 10

def predict(image, net):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY), colors[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    return image

def capture_predict():
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
    os.system("fswebcam -r 299x299 --jpeg 85 -S 20 teste.jpg")
    image = cv2.imread('teste.jpg')
    output = predict(image, net)
    cv2.imwrite("output.jpg", output)
    cv2.imshow("Output", output)
    cv2.waitKey(0)

# rotate 300ms right per frame
# 4500ms = ~360dg
def search(obj):
    print 'searching for a', obj

def forward(msecs):
    arduin_com.write('1')
    arduin_com.write(str(round_msecs(msecs)))

def turn_right(msecs):
    units = int(round(float(msecs) / msecs_per_unit))
    while units > 0:
        arduin_com.write('2' + str(min(units-1,9)))
        units -= min(units, 10)
    time.sleep((msecs * 1.2) / 1000.0)

def turn_left(msecs):
    units = int(round(float(msecs) / msecs_per_unit))
    while units > 0:
        arduin_com.write('3' + str(min(units-1,9)))
        units -= min(units, 10)
    time.sleep((msecs * 1.2) / 1000.0)

def backward(msecs):
    arduin_com.write('4')
    arduin_com.write(str(round_msecs(msecs)))

def beep(times):
    arduin_com.write('5')
    arduin_com.write(str(times))

def receive(rcv_socket):
    data = rcv_socket.recv(2048)
    cmd = list(str(data.decode('utf-8')).lower())
    cmd = ''.join([c for c in cmd if c != '\x00'])
    words = cmd.split(' ')
    if 'encontre' in words or 'pegue' in words:
        obj = words[2]
        if obj in objects:
            search(objects[obj])
    rcv_socket.send('ack'.encode(encoding='utf-8', errors='ignore'))
    rcv_socket.close()

for i in range(0, 13):
    os.system("fswebcam -r 299x299 --jpeg 85 -S 10 img"+str(i)+".jpg")
    turn_right(300)

'''
while True:
    msecs = int(raw_input())
    if msecs > 0:    
        turn_left(msecs)
    else:
        turn_right(msecs * -1)
'''
'''my_socket = socket(AF_INET,SOCK_STREAM)
my_socket.bind(('',socket_num))
my_socket.listen(1)

while True:
    rcv_socket, addr = my_socket.accept()
    new_thread = threading.Thread(target=receive, args=(rcv_socket,))
    new_thread.start()
my_socket.close()'''
