import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
from joblib import load, dump


#https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def contrastLab(img_in_bgr):
    img_lab= cv2.cvtColor(img_in_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe_stuff = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe_stuff.apply(l)

    limg = cv2.merge((cl,a,b))


    contrasted_lab_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return contrasted_lab_img
# kod sa prvog izazova
def enhance_gray(img_bgr):
    img_contrast = contrastLab(img_bgr)
    img = increase_brightness(img_contrast)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_gs = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gs, (5, 5), 0)
    img_gs = 0-blur
    return img_gs

def enhance_gray_gender(img_bgr):
    img_contrast = contrastLab(img_bgr)
    img = increase_brightness(img_contrast)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_gs = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gs = 0-img_gs
    return img_gs


# https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
def increase_brightness(img, value=45):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# kod sa vezbi
def detect_face(img_path):
    # inicijalizaclija dlib detektora (HOG)
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    img_bgr = cv2.imread(img_path)
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


    # detekcija svih lica na grayscale slici

    rects = detector(image2, 1)

    # iteriramo kroz sve detekcije korak 1.
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        # odredjivanje kljucnih tacaka - korak 2
        shape = predictor(image2, rect)
        # shape predstavlja 68 koordinata

        shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
        # print("Dimenzije prediktor matrice: {0}".format(shape.shape))  # 68 tacaka (x,y)
        # print("Prva 3 elementa matrice")
        # print(shape[:3])

        j = 0
        for s in shape[:68]:
            if s[0] < 0:
                shape[j] = [0, s[1]]

            j = j + 1

        # konvertovanje pravougaonika u bounding box koorinate
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # crtanje pravougaonika oko detektovanog lica
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rectangle_coords = [x, y, x + w, y + h]

        if x < 0:
            rectangle_coords[0] = 0
        if y < 0:
            rectangle_coords[1] = 0

        if x + w > image.shape[1]:
            rectangle_coords[2] = image.shape[1]

        if y + h > image.shape[0]:
            rectangle_coords[3] = image.shape[0]

        # crtanje kljucnih tacaka
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        for (x, y) in shape:
            if coords_dont_exist_in_rect(rectangle_coords, x, y):
                rectangle_coords = expand_or_shrink_rectangle(rectangle_coords, x, y)

        cv2.rectangle(image2, (rectangle_coords[0], rectangle_coords[1]), (rectangle_coords[2], rectangle_coords[3]),
                      (0, 255, 0), 2)

        face = image2[rectangle_coords[1] + 2:rectangle_coords[3] - 2, rectangle_coords[0] + 2: rectangle_coords[2] - 2]


        return face


def coords_dont_exist_in_rect(rectangle_coords, x, y):
    if x >= rectangle_coords[0] and x <= rectangle_coords[2]:
        if y >= rectangle_coords[1] and y <= rectangle_coords[3]:
            return False
    return True


def expand_or_shrink_rectangle(rectangle_coords, x, y):
    if x < rectangle_coords[0]:
        rectangle_coords[0] = x

    if x > rectangle_coords[2]:
        rectangle_coords[2] = x

    if y < rectangle_coords[1]:
        rectangle_coords[1] = y

    if y > rectangle_coords[3]:
        rectangle_coords[3] = y

    return rectangle_coords


def resize_img(img):
    return cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)

