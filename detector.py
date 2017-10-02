import numpy as np
import argparse
import cv2
import serial
import threading
import os
from socket import *
from PIL import Image

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
socket_num = 3001
arduin_com = serial.Serial('/dev/ttyACM0', 9600)

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
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
    os.system("fswebcam -r 299x299 --jpeg 85 -S 20 teste.jpg")
    image = cv2.imread('teste.jpg')
    output = predict(image, net)
    cv2.imwrite("output.jpg", output)
    cv2.imshow("Output", output)
    cv2.waitKey(0)

def receive(rcv_socket):
    data = rcv_socket.recv(2048)
    cmd = list(str(data.decode('utf-8')).lower())
    cmd = ''.join([c for c in cmd if c != '\x00'])
    print (cmd)
    if cmd == 'va para frente':
	arduin_com.write('1')
	arduin_com.write('4')
    elif cmd == 'vire para direita':
	arduin_com.write('2')
	arduin_com.write('2')
    elif cmd == 'vire para esquerda':
	arduin_com.write('3')
	arduin_com.write('2')
    elif cmd == 'va para tras':
	arduin_com.write('4')
	arduin_com.write('2')
    elif cmd == 'faca barulho':
        arduin_com.write('5')
	arduin_com.write('3')
    elif cmd == 'mostre todas as funcionalidades':
        arduin_com.write('1')
	arduin_com.write('2')
        arduin_com.write('2')
	arduin_com.write('2')
        arduin_com.write('3')
	arduin_com.write('2')
        arduin_com.write('4')
	arduin_com.write('2')
        arduin_com.write('5')
	arduin_com.write('3')
    rcv_socket.send('ack'.encode(encoding='utf-8', errors='ignore'))
    rcv_socket.close()

my_socket = socket(AF_INET,SOCK_STREAM)
my_socket.bind(('',socket_num))
my_socket.listen(1)

while True:
    rcv_socket, addr = my_socket.accept()
    new_thread = threading.Thread(target=receive, args=(rcv_socket,))
    new_thread.start()
my_socket.close()