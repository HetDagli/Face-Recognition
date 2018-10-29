from sklearn import svm
import dlib
fl=open("/home/shubham/Documents/project/Final_project/data_functioning/data_images.txt","r")
lst_data=[]
lst_label=[]
lst_label_name=[]
for line in fl:
    if(line!=""):
        str_arr,label=tuple(line.split(";"))
        label,label_id=tuple(label.split(":"))
        arr_com=list(map(float,str_arr.split(",")))
        lst_data.append(arr_com)
        lst_label_name.append(label)
        lst_label.append(int(label_id))
x=dlib.vectors()
y=dlib.vectors()
for i in range(0,len(lst_label)):
    x.append(dlib.vector(lst_data[i]))
    y.append(dlib.vector([lst_label[i]]))
svm=dlib.svm_c_trainer_linear()
svm.be_verbose()
svm.set_c(10)
classifier=svm.train(x,y)

print("Prediction for first sample: "+lst_label_name[classifier(x[0])-1])
