import pickle
pickle_clf=open("classifier.pickle","rb")
clf=pickle.load(pickle_clf)
def match_name(arr):
    return str(clf.predict([arr]))
