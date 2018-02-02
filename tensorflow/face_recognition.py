from math import sqrt
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
import numpy as np


# load data from file
def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


# train knn_model
def train(train_dir, model_save_path = "", n_neighbors = None, knn_algo = 'ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model of disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified.
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        for data_path in listdir(join(train_dir, class_dir)):
            X.append(load_data(join(join(train_dir, class_dir), data_path)))
            y.append(class_dir)


    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically as:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path != "":
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf


# predict
def predict(frame, knn_clf = None, model_save_path ="", DIST_THRESH = .3):
    """
    recognizes faces in given image, based on a trained knn classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_save_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param DIST_THRESH: (optional) distance threshold in knn classification. the larger it is, the more chance of misclassifying an unknown person to a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'N/A' will be passed.
    """

    # if not isfile(X_img_path) or splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #     raise Exception("invalid image path: {}".format(X_img_path))
    #
    # if knn_clf is None and model_save_path == "":
    #     raise Exception("must supply knn classifier either thourgh knn_clf or model_save_path")
    #
    # if knn_clf is None:
    #     with open(model_save_path, 'rb') as f:
    #         knn_clf = pickle.load(f)

    # X_img = face_recognition.load_image_file(X_img_path)
    X_faces_loc = face_locations(frame, model='hog')
    if len(X_faces_loc) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(frame, known_face_locations=X_faces_loc)


    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]

    # predict classes and cull classifications that are not with high confidence
    return [(pred, loc) if rec else ("N/A", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_faces_loc, is_recognized)]



knn_model = train("./train")
result = dict()


# face
def face_rec(frame, frame_name):
    preds = predict(frame, knn_clf=knn_model)
    for pred in preds:
        loc = pred[1]
        loc_ = loc + (frame_name,)
        name = pred[0]
        if name != 'N/A':
            if result.get(name, '?') == '?':
                result[name] = loc_
    return result