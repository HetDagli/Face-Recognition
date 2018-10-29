import cv2
import dlib
from time import sleep
import pickle
predictor_url="data_functioning/shape_predictor_68_face_landmarks.dat"
recog_model_url="data_functioning/dlib_face_recognition_resnet_model_v1.dat"
to_recog="data/"
detector=dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1("/home/shubham/Documents/project/Final_project/data_functioning/mmod_human_face_detector.dat")
sp=dlib.shape_predictor(predictor_url)
facerec=dlib.face_recognition_model_v1(recog_model_url)
lst_vals=[]
dict_vals={}
pickle_clf=open("classifier.pickle","rb")
clf=pickle.load(pickle_clf)
fl_data=open("data_txt.txt","r")
data_arrs=[]
data_arrs_labels=[]
#win = dlib.image_window()
for line in fl_data:
    act_data=line.strip("\n")
    data_arr,label_name=act_data.split(";")
    data_final_arr=list(map(float,data_arr.split(",")))
    data_arrs.append(data_final_arr)
    data_arrs_labels.append(label_name)
#print(data_arrs_labels)
#print(data_arrs)
def euclidean_dist(arr,arr_com):
    return sum((float(a)-float(b))**2 for a,b in zip(arr_com,arr))**0.5
def match_name(arr):
    for i in range(0,len(data_arrs)):
        if(euclidean_dist(arr,data_arrs[i])<0.5):
            #print("Euclidean distance predicted Name: "+data_arrs_labels[i])
            #print("Classifier predicted Name: "+str(clf.predict([arr])))
            return True
    return False
global last_key
last_key=0
def euclidean_dist(arr,arr_com):
    return sum((float(a)-float(b))**2 for a,b in zip(arr_com,arr))**0.5
global population
population=0
def add_to_dict(arr):
    global population
    global last_key

    if(last_key!=0):
        for key in range(1,last_key):
            if(euclidean_dist(arr,dict_vals[key][0])<0.6):
                dict_vals[key].append(arr)
                return False
        dict_vals[last_key+1]=[arr]
        last_key=last_key+1
        return True
    else:
        dict_vals[last_key+1]=[arr]
        last_key=last_key+1
        return True
def process_frame(frame):
    #Returns the array of all the faces detected
    dets=detector(frame,1)
    #win.clear_overlay()
    #win.set_image(frame)
    #dets = cnn_face_detector(frame, 1)
    if len(dets)!=0:
        for l,d in enumerate(dets):
            shape=sp(frame,d)
            #win.clear_overlay()
            #win.add_overlay(d)
            #win.add_overlay(shape)
            face_descriptor=facerec.compute_face_descriptor(frame,shape)
            arr=list(face_descriptor)
            if(add_to_dict(arr)):
                match_name(arr)
cap=cv2.VideoCapture("process_vdo.mp4")
while cap.isOpened():
    ret,frame=cap.read()
    #frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret==True:
        #num_rows, num_cols = frame.shape[:2]
        #rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
        #frame = cv2.warpAffine(frame, rotation_matrix, (num_cols, num_rows))
        #cv2.imshow('Video',frame)
        process_frame(frame)
        if(cv2.waitKey(10) & 0xFF == ord('q')):
            break
    else:
        break
print(len(dict_vals.keys()))
cap.release()
cv2.destroyAllWindows()
