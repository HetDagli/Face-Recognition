import numpy as np
from sklearn import svm
import pickle
import glob
import os
import dlib
import cv2
##Final Variables
predictor_url="/home/shubham/Documents/project/Final_project/data_functioning/shape_predictor_68_face_landmarks.dat"
recog_model_url="/home/shubham/Documents/project/Final_project/data_functioning/dlib_face_recognition_resnet_model_v1.dat"
to_recog="/home/shubham/Documents/project/Final_project/data/"
detector=dlib.get_frontal_face_detector()
sp=dlib.shape_predictor(predictor_url)
facerec=dlib.face_recognition_model_v1(recog_model_url)
files=[]
def get_arr(frame):
    dets=detector(frame,1)
    if len(dets)!=0:
        for l,d in enumerate(dets):
            shape=sp(frame,d)
            face_descriptor=facerec.compute_face_descriptor(frame,shape)
            return list(face_descriptor)
def arr_to_str(arr):
    retStr=str(arr[0])
    for i in range(1,len(arr)):
        retStr=retStr+","+str(arr[i])
    return retStr
for f in glob.glob(os.path.join("training_vdos/", "*.mp4")):
    files.append(f)
for fl_url in files:
    label_processing=fl_url.split("/")[1].split(".")[0]
    cap=cv2.VideoCapture(fl_url)
    while cap.isOpened():
        ret,frame=cap.read()
        if ret==True:
            cv2.imshow("Image",frame)
            if(cv2.waitKey(10) & 0xFF == ord('s')):
                fl_data=open("data_txt.txt","a")
                fl_data.write(arr_to_str(get_arr(frame))+";"+label_processing)
                fl_data.write("\n")
                break
    cap.release()
    cv2.destroyAllWindows()
    print("Done Processing: ",fl_url)
