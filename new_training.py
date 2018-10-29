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
detector=dlib.get_frontal_face_detector()
sp=dlib.shape_predictor(predictor_url)
facerec=dlib.face_recognition_model_v1(recog_model_url)
win=dlib.image_window()
for f in glob.glob(os.path.join(to_recog,"*.jpg")):
    print("Processing file: {}".format(f))
    img=cv2.imread(f)
    win.clear_overlay()
    win.set_image(img)
    dets=detector(img,1)
    print(dets)
    print("Number of faces detected: {}".format(len(dets)))
    count=0
    for k,d in enumerate(dets):
        print("Detecting {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k,d.left(),d.top(),d.right(),d.bottom()))
        shape=sp(img,d)
        ##win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)
        #label_name=raw_input("Enter name of person: ")
        face_descriptor=facerec.compute_face_descriptor(img,shape)
        arr=[]
        for val in face_descriptor:
            arr.append(val)
        count+=1
print(count)
##        fl=open("/home/shubham/Documents/project/Final_project/data_functioning/data_images.txt","a")
##        fl.write(str(arr[0]))
##        for x in range(1,len(arr)):
##            fl.write(","+str(arr[x]))
##        lastcount=lastcount+1
##        fl.write(";"+label_name+":"+str(lastcount))
##        fl.write("\n")
##        fl.close()
