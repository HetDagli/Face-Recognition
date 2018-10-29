import dlib
import sys
import os
import glob
import cv2
predictor_url="shape_predictor_68_face_landmarks.dat"
recog_model_url="dlib_face_recognition_resnet_model_v1.dat"
to_recog="data/"
label_name=raw_input("Enter name of person: ")
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
    for k,d in enumerate(dets):
        print("Detecting {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k,d.left(),d.top(),d.right(),d.bottom()))
        shape=sp(img,d)
        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)
    face_descriptor=facerec.compute_face_descriptor(img,shape)
    arr=[]
    for val in face_descriptor:
        arr.append(val)
"""fl=open("data.txt","r")
for line in fl:
    arr_com=line.split(",")
print((sum((float(a)-b)**2 for a,b in zip(arr_com,arr)))**0.5)"""
fl=open("/home/shubham/Documents/project/Final_project/data_images.txt","a")
fl.write(str(arr[0]))
for x in range(1,len(arr)):
    fl.write(","+str(arr[x]))
fl.write(";"+label_name)
fl.write("\n")
print("All completed")
fl.close()
