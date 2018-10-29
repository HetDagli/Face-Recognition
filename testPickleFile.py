import pickle
from sklearn import svm
data_clf=open("data.pickle","rb")
data=pickle.load(data_clf)
clf=svm.SVC(kernel="linear",gamma=0.001,C=100)
clf.fit(data[0],data[1],sample_weight=None)
with open("classifier.pickle","wb") as clf_pkl:
    pickle.dump(clf,clf_pkl)
