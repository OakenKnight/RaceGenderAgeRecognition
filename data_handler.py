import cv2
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler as ros
from hog import create_hog
import image_handler as ih
def classify_gender(train_image_paths, train_image_labels):
    men=[]
    women=[]
    for i in range(len(train_image_paths)):
        if train_image_labels[i] == '1':
            women.append(train_image_paths[i])
        else:
            men.append(train_image_paths[i])

    return women,men


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


def reshape_data_for_test(input_data):
    nx, ny = input_data.shape
    array = input_data.reshape((nx*ny))
    return array.reshape(1,-1)

def classify_race(train_image_paths, train_image_labels):
    white=[]
    negro=[]
    asian = []
    indian = []
    others=[]

    for i in range(len(train_image_paths)):
        if train_image_labels[i] == '0':
            white.append(train_image_paths[i])
        elif train_image_labels[i] == '1':
            negro.append(train_image_paths[i])
        elif train_image_labels[i] == '2':
            asian.append(train_image_paths[i])
        elif train_image_labels[i] == '3':
            indian.append(train_image_paths[i])
        else:
            others.append(train_image_paths[i])

    print("w: ", len(white), " n: ", len(negro), " a: ", len(asian), " i: ",len(indian)," o: ",len(others))
    return white,negro,asian,indian,others

