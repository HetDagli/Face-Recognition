import cv2
from PyQt4 import QtGui, QtCore
import dlib
import sys
import os
import glob
import cv2
import threading
import datetime
import match_name
import pickle
present_lst=[]
def euclidean_dist(arr,arr_com):
    return sum((float(a)-float(b))**2 for a,b in zip(arr_com,arr))**0.5
##def match_name(arr):
##    fl=open("/home/shubham/Documents/project/Final_project/data_functioning/data_images.txt","r")
##    dist_min=0.4
##    label_min=""
##    id_min="0"
##    for line in fl:
##        if(line!=""):
##            str_arr,label=tuple(line.split(";"))
##            label,label_id=tuple(label.split(":"))
##            arr_com=list(map(float,str_arr.split(",")))
##            dist=euclidean_dist(arr,arr_com)
##            if(dist<dist_min):
##                dist_min=dist
##                label_min=label
##                id_min=label_id
##    if int(id_min) not in present_lst:
##        present_lst.append(int(id_min))
##    return label_min
pickle_clf=open("classifier.pickle","rb")
clf=pickle.load(pickle_clf)
def match_name(arr):
    ret_val=str(clf.predict([arr]))
    print(ret_val)
    return ret_val
class Capture():
    def __init__(self,window_object):
        self.capturing = False
        self.c = cv2.VideoCapture(0)
        self.predictor_url="/home/shubham/Documents/project/Final_project/data_functioning/shape_predictor_68_face_landmarks.dat"
        self.recog_model_url="/home/shubham/Documents/project/Final_project/data_functioning/dlib_face_recognition_resnet_model_v1.dat"
        self.to_recog="/home/shubham/Documents/project/eigenFaces/DeepLearning/Data/"
        self.detector=dlib.get_frontal_face_detector()
        self.sp=dlib.shape_predictor(self.predictor_url)
        self.facerec=dlib.face_recognition_model_v1(self.recog_model_url)
        self.image=None
        self.window_object=window_object
    def face_recognize(self):
        if(self.image is not None):
            arr=[]
            face_descriptor=None
            dets=self.detector(self.image,1)
            for k,d in enumerate(dets):
                shape=self.sp(self.image,d)
                face_descriptor=self.facerec.compute_face_descriptor(self.image,shape)
            if(face_descriptor is not None):
                arr=list(face_descriptor)
            if(len(arr)!=0):
                label_obtained=match_name(arr)
                if(label_obtained==""):
                    self.window_object.label_name.setText("No match found.")
                else:
                    self.window_object.label_name.setText("Match found: "+label_obtained)
        threading.Timer(1, self.face_recognize).start()
    def startCapture(self):
        self.face_recognize()
        self.capturing = True
        cap = self.c
        while(self.capturing):
            ret, frame = cap.read()
            cv2.imshow("Capture", frame)
            self.image=frame
            cv2.waitKey(5)
        cv2.destroyAllWindows()

    def endCapture(self):
        self.capturing = False

    def quitCapture(self):
        cap = self.c
        cv2.destroyAllWindows()
        cap.release()
        self.capturing = False
        sys.exit()
        QtCore.QCoreApplication.quit()
def save_attendance(win):
    now=datetime.datetime.now()
    #time in dd-mm-yyyy:lectureId format is the filename
    str_lid=win.enter_lectureId.text()
    str_filename="/home/shubham/Documents/project/Final_project/attendance_files/"+str(now.day)+"-"+str(now.month)+"-"+str(now.year)+":"+str_lid+".csv"
    fl_lastcount=open("/home/shubham/Documents/project/Final_project/data_functioning/save_lastcount.txt","r")
    for line in fl_lastcount:
        last_num=int(line)
    fl_lastcount.close()
    fl_csv=open(str_filename,"w")
    fl_csv.write("ID,STATUS")
    fl_csv.write("\n")
    for i in range(1,last_num+1):
        if i in present_lst:
            fl_csv.write(str(i)+","+"present")
        else:
            fl_csv.write(str(i)+","+"absent")
        fl_csv.write("\n")
    fl_csv.close()
    os._exit(0)
class Window(QtGui.QWidget):
    def __init__(self):

        QtGui.QWidget.__init__(self)
        self.setWindowTitle('Control Panel')
        ##self.showFullScreen()
        self.setStyleSheet("background-color: black;")
        self.capture = Capture(self)
        self.start_button = QtGui.QPushButton('Start',self)
        self.start_button.clicked.connect(self.capture.startCapture)

        self.end_button = QtGui.QPushButton('End',self)
        self.end_button.clicked.connect(self.capture.endCapture)
        self.label_li=QtGui.QLabel()
        self.label_li.setText("Enter Lecture Id")
        self.enter_lectureId=QtGui.QLineEdit(self)
        self.quit_button = QtGui.QPushButton('Quit',self)
        self.quit_button.clicked.connect(self.capture.quitCapture)
        self.label_name=QtGui.QLabel()
        self.label_name.setText("Finding match.")
        self.save_button = QtGui.QPushButton('Save',self)
        self.save_button.clicked.connect(lambda: save_attendance(self))
        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.label_li)
        vbox.addWidget(self.enter_lectureId)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.quit_button)
        vbox.addWidget(self.save_button)
        vbox.addWidget(self.label_name)
        self.setLayout(vbox)
        self.setGeometry(100,100,400,300)
        self.show()

if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
