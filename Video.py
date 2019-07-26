#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 22:03:24 2019

@author: CristianHernandez
"""

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read("trainner.yml")

cap = cv2.VideoCapture(0)
# Define el codec y crea el objeto VideoWriter
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(True):
    ret , frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for(x, y, w, h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf <= 85:
            print(id_)
        
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        cv2.putText(frame, 'Julian Marquez', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), lineType=cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#Libera todo si la tarea ha terminado
cap.release()
cv2.destroyAllWindows()