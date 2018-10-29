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
global label_processing
files=[]
data_arr=[]
data_target=[]
for f in glob.glob(os.path.join("training_vdos/", "*.mp4")):
    files.append(f)
def add_to_list(arr,fl_url):
    data_arr.append(arr)
    data_target.append(fl_url)
def process_frame(frame,fl_url):
    #Returns the array of all the faces detected
    dets=detector(frame,1)
    if len(dets)!=0:
        for l,d in enumerate(dets):
            shape=sp(frame,d)
            face_descriptor=facerec.compute_face_descriptor(frame,shape)
            arr=list(face_descriptor)
            add_to_list(arr,fl_url)
for fl_url in files:
    count=1
    num_images=0
    check=True
    label_processing=fl_url.split("/")[1].split(".")[0]
    cap=cv2.VideoCapture(fl_url)
    while cap.isOpened():
        ret,frame=cap.read()
        check=True
        if(count%5!=0):
            check==False
        count=count+1
        if ret==True:
            if(check==True):
                num_images+=1
                process_frame(frame,label_processing)
        else:
            break
    #print("Done processing: %s".format(fl_url))
    print(num_images)
    cap.release()
    cv2.destroyAllWindows()
    print("Done Processing: ",fl_url)
#clf=svm.SVC(kernel="linear",gamma=0.001,C=100)
#clf.fit(data_arr,data_target,sample_weight=None)
data_final=(data_arr,data_target)
with open("data.pickle","wb") as d_pkl:
    pickle.dump(data_final,d_pkl)
#with open("classifier.pickle","wb") as clf_pkl:
    #pickle.dump(clf,clf_pkl)
