import socket
import numpy as np
import argparse
import cv2 
import time
 
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                  help='Path to text network fie: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model or '
                                       )
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                      )
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()


classNames = { 0: 'fondo',
    1: 'avion', 2: 'bicicleta', 3: 'ave', 4: 'serpiente',
    5: 'bote', 6: 'autobus', 7: 'car', 8: 'gato', 9: 'silla',
    10: 'vaca', 11: 'no reconocido', 12: 'perro', 13: 'caballo',
    14: 'motocicleta', 15: 'persona', 16: 'planta',
    17: 'borrega', 18: 'sofa', 19: 'tren', 20: 'monitor' }

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(("192.168.1.102", 5000))
 
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture('http://192.168.1.102:8081')


net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

while True:
    #s.connect(("192.168.1.234", 5000))
   
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

    
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    
    net.setInput(blob)
    
    detections = net.forward()

    
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > args.thr: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label

             
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
            
            
            heightFactor = frame.shape[0]/300.0  
            widthFactor = frame.shape[1]/300.0 
            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)
              
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))

            
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                if confidence == None:
                        mensaje =str("start")
                if  confidence < .6:
                        mensaje=str("start")
                elif confidence > .6:
                        mensaje="stop"
                s.send(mensaje.encode('ascii'))
                #s.close()
                print(label)
                mensaje=" "    
    cv2.namedWindow("aver", cv2.WINDOW_NORMAL)
    cv2.imshow("aver", frame)
    if cv2.waitKey(1) >= 0:  
        break
#s.connect(("192.168.1.234", 5000))
s.close()

