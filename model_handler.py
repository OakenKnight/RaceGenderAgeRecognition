import os
from joblib import load,dump
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR

def model_exists(model_name):
    if os.path.exists(model_name):
        return True
    return False

def load_model(model_name):
    if model_exists(model_name):
        try:
            clf_knn = load(model_name)
            return clf_knn
        except Exception as e:
            print("Warning: No model found!")
            return None

def create_model_race(x_train,y_train, model_name):
    clf_svm = SVC(kernel='linear', probability=True)

    clf_svm.fit(x_train, y_train)

    clf_knn = KNeighborsClassifier(n_neighbors=10)
    clf_knn = clf_knn.fit(x_train, y_train)

    dump(clf_knn, model_name)
    print("Created " + model_name)

    return clf_knn

def create_model_gender(x_train,y_train, model_name):
    clf_svm = SVC()
    clf_svm.fit(x_train, y_train)

    clf_knn = KNeighborsClassifier(n_neighbors=10, )
    clf_knn = clf_knn.fit(x_train, y_train)

    dump(clf_knn, model_name)
    print("Created " + model_name)

    return clf_knn



#https://medium.com/coinmonks/support-vector-regression-or-svr-8eb3acf6d0ff
#https://towardsdatascience.com/machine-learning-basics-support-vector-regression-660306ac5226
def create_model_age(x_train,y_train, model_name):

    regressor = SVR(kernel='rbf',epsilon=1.0,degree=5)
    regressor.fit(x_train, y_train)

    dump(regressor, model_name)
    print("Created " + model_name)

    return regressor

def handle_model(x_train,y_train, model_name):
    if load_model(model_name) is not None:
        print("Loading "+ model_name)
        return load_model(model_name)
    else:
        if model_name == "knn_race.joblib":
            print("Creating "+model_name)
            return create_model_race(x_train,y_train,model_name)
        elif model_name=="knn_gender.joblib":
            print("Creating "+model_name)
            return create_model_gender(x_train,y_train,model_name)
        elif model_name=="knn_age.joblib":
            print("Creating "+model_name)
            return create_model_age(x_train,y_train,model_name)
