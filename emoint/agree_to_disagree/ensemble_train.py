import pickle



from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import *
from sklearn import svm
from sklearn.svm import SVC

import xgboost as xgb
import numpy as np


if __name__ == '__main__':
    clf = AdaBoostClassifier(xgb.XGBClassifier(nthreads=-1, n_estimators=100))
    print("Data Started")
    train_data = pickle.load(open('/home/venkatesh/pkl_dir/ensemble_train_data.pkl', 'rb'))
    dev_data = pickle.load(open('/home/venkatesh/pkl_dir/ensemble_dev_data.pkl', 'rb'))
    test_data = pickle.load(open('/home/venkatesh/pkl_dir/ensemble_test_data.pkl', 'rb'))
    print("Data Loaded")
    
    np.random.seed(0)
    # clf = RandomForestClassifier(n_jobs=-1, n_estimators=100)
    # clf = svm.SVC()
    # clf = SVC(kernel="linear", C=0.025)
    # clf = xgb.XGBClassifier(nthread=-1)

    clf.fit(train_data[0][:, :-1], train_data[1])
    dev_pred = clf.predict(dev_data[0][:, :-1])
    test_pred = clf.predict(test_data[0][:, :-1])

    print(accuracy_score(dev_data[1], dev_pred), "Accuracy")
    print(accuracy_score(test_data[1], test_pred), "Accuracy")
    print(f1_score(test_data[1], test_pred, average='weighted'), "F1 Score")
    print(f1_score(dev_data[1], dev_pred, average='weighted'), "F1 Score")



