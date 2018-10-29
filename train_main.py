import dlib
import sys
import os
import glob
import cv2
fl_lastcount=open("/home/shubham/Documents/project/Final_project/data_functioning/save_lastcount.txt","r")
lastcount=0
for line in fl_lastcount:
    lastcount=int(line)
predictor_url="/home/shubham/Documents/project/Final_project/data_functioning/shape_predictor_68_face_landmarks.dat"
recog_model_url="/home/shubham/Documents/project/Final_project/data_functioning/dlib_face_recognition_resnet_model_v1.dat"
to_recog="/home/shubham/Documents/project/Final_project/data/"
label_name=raw_input("Enter name of person: ")
detector=dlib.get_frontal_face_detector()
sp=dlib.shape_predictor(predictor_url)
facerec=dlib.face_recognition_model_v1(recog_model_url)
cap=cv2.VideoCapture(0)
lastcount=lastcount+1
while(True):
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image",gray)
    k=cv2.waitKey(1) & 0xff
    if k==13:
        dets=detector(gray,1)
        for l,d in enumerate(dets):
            shape=sp(img,d)
            face_descriptor=facerec.compute_face_descriptor(img,shape)
        arr=list(face_descriptor)
        print(len(arr))
        fl=open("/home/shubham/Documents/project/Final_project/data_functioning/data_images.txt","a")
        fl.write(str(arr[0]))
        for x in range(1,len(arr)):
            fl.write(","+str(arr[x]))
        fl.write(";"+label_name+":"+str(lastcount))
        fl.write("\n")
        fl.close()  
        fl_lastcount=open("/home/shubham/Documents/project/Final_project/data_functioning/save_lastcount.txt","w")
        fl_lastcount.write(str(lastcount))
        fl_lastcount.close()
        print("All completed")

