import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils
from joblib import load
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.preprocessing import LabelEncoder

import data_handler as dh
import image_handler as ih
from model_handler import handle_model
from hog import create_hog

def train_or_load_age_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju
    age_features=[]
    img = cv2.imread(train_image_paths[0])

    hog = create_hog(img)
    labels = []
    samples=[]
    """
    i=0
    for img_path in train_image_paths:
        samples.append([img_path,train_image_labels[i]])
        i +=1

    sorted_samples = sorted(samples,key=lambda tup:tup[1])

    for sample in sorted_samples:
        img = cv2.imread(sample[0])
        gs = ih.enhance_gray(img)
        age_features.append(hog.compute(gs))
        labels.append(sample[1])
    
    """
    """
     i=0
    for img_path in train_image_paths:
        img = cv2.imread(img_path)
        gs = ih.enhance_gray(img)
        age_features.append(hog.compute(gs))
        labels.append(train_image_labels[i])
        i +=1

    
    
    """
    i = 0
    for img_path in train_image_paths:
        samples.append([img_path, train_image_labels[i]])
        i += 1

    #sorted_samples = sorted(samples, key=lambda tup: tup[1])

    for sample in samples:
        img = cv2.imread(sample[0])
        gs = ih.enhance_gray(img)
        age_features.append(hog.compute(gs))
        labels.append(sample[1])



    age_features = np.array(age_features)
    y_train = np.array(labels)

    x_train = dh.reshape_data(age_features)

    clf_knn = handle_model(x_train,y_train,"knn_age.joblib")


    return clf_knn


def train_or_load_gender_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    women_imgs, men_imgs = dh.classify_gender(train_image_paths, train_image_labels)

    women_features = []
    men_features = []
    labels = []

    img = cv2.imread(train_image_paths[0])

    hog = create_hog(img)

    for img_path in women_imgs:
        img = cv2.imread(img_path)
        gs = ih.enhance_gray_gender(img)
        img2 = cv2.flip(img,1)
        gs2 = ih.enhance_gray_gender(img2)

        women_features.append(hog.compute(gs))
        #women_features.append(hog.compute(gs2))
        labels.append(1)
        #labels.append(1)

    for img_path in men_imgs:
        img = cv2.imread(img_path)
        gs = ih.enhance_gray_gender(img)
        img2=cv2.flip(img,1)
        gs2 = ih.enhance_gray_gender(img2)

        men_features.append(hog.compute(gs))
        #men_features.append(hog.compute(gs2))
        labels.append(0)
        #labels.append(0)

    women_features = np.array(women_features)
    men_features = np.array(men_features)

    x_train = np.vstack((women_features, men_features))
    y_train = np.array(labels)

    x_train = dh.reshape_data(x_train)


    clf_knn = handle_model(x_train,y_train,"knn_gender.joblib")

    return clf_knn


def train_or_load_race_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    whites,niggas,asians, indians,others = dh.classify_race(train_image_paths, train_image_labels)

    white_features = []
    nigga_features = []
    asian_features = []
    indian_features=[]
    other_features=[]
    labels = []

    img = cv2.imread(train_image_paths[0])

    hog = create_hog(img)

    for img_path in whites:
        img = cv2.imread(img_path)
        gs = ih.enhance_gray(img)
        white_features.append(hog.compute(gs))
        labels.append(0)

    for img_path in niggas:
        img = cv2.imread(img_path)
        gs = ih.enhance_gray(img)

        nigga_features.append(hog.compute(gs))
        labels.append(1)

    for img_path in asians:
        img = cv2.imread(img_path)
        gs = ih.enhance_gray(img)
        asian_features.append(hog.compute(gs))
        labels.append(2)

    for img_path in indians:
        img = cv2.imread(img_path)
        gs = ih.enhance_gray(img)
        indian_features.append(hog.compute(gs))
        labels.append(3)

    for img_path in others:
        img = cv2.imread(img_path)
        gs = ih.enhance_gray(img)
        other_features.append(hog.compute(gs))

        labels.append(4)

    white_features = np.array(white_features)
    nigga_features = np.array(nigga_features)
    asian_features = np.array(asian_features)
    indian_features = np.array(indian_features)
    other_features = np.array(other_features)

    x_train = np.vstack((white_features, nigga_features,asian_features,indian_features, other_features))
    y_train = np.array(labels)
    #y_train = LabelEncoder().fit_transform(y_train)
    x_train = dh.reshape_data(x_train)
    #smt = SMOTE(random_state=None, sampling_strategy='not minority')

    #x_train, y_train = smt.fit_resample(x_train,y_train)
    clf_knn = handle_model(x_train,y_train,'knn_race.joblib')

    return clf_knn


def predict_age(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje godina i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati godine.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje godina
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati godine lica
    :return: <Int> Prediktovanu vrednost za goinde  od 0 do 116
    """
    gs_face = ih.detect_face(image_path)
    resized_img = ih.resize_img(gs_face)
    hog = create_hog(resized_img)
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    x_test = hog.compute(resized_img)
    x_test = np.array(x_test)

    x_test = dh.reshape_data_for_test(x_test)


    age = trained_model.predict(x_test)
    age = int(age)
    print(image_path)
    print(age)
    return age




def predict_gender(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje pola na osnovu lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati pol.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa pola (0 - musko, 1 - zensko)
    """
    # print(image_path)
    gs_face = ih.detect_face(image_path)
    resized_img = ih.resize_img(gs_face)
    hog = create_hog(resized_img)
    gender = 1
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    x_test = hog.compute(resized_img)
    x_test = np.array(x_test)

    x_test = dh.reshape_data_for_test(x_test)

    # print('Test shape: ', x_test.shape)

    gender = trained_model.predict(x_test)

    gender = int(gender)

    print(gender)
    return gender


def predict_race(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje rase lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati rasu.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <Int>  Prepoznata klasa (0 - Bela, 1 - Crna, 2 - Azijati, 3- Indijci, 4 - Ostali)
    """
    gs_face = ih.detect_face(image_path)
    resized_img = ih.resize_img(gs_face)
    hog = create_hog(resized_img)

    race = 4

    x_test = hog.compute(resized_img)
    x_test = np.array(x_test)

    x_test = dh.reshape_data_for_test(x_test)

    race = trained_model.predict(x_test)
    race = int(race)
    print(race)
    return race
