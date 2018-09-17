import cv2
import dlib
import imutils
import time
import numpy as np
import openface
import pickle
from pyspark import SparkContext
from skimage import io, feature
from pyspark.sql import SQLContext, Row

# load Models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_detector = dlib.get_frontal_face_detector()
predictor_model = "shape_predictor_68_face_landmarks.dat"
face_aligner = openface.AlignDlib(predictor_model)
with open('./TrainSVM/feature/classifier.pkl', 'rb') as f:
    (le, clf) = pickle.load(f)

def FaceDetection(gray):
    detected_faces = face_cascade.detectMultiScale(gray)
    return detected_faces

def FaceDetection1(gray):
    detected_faces = face_detector(gray, 0)
    return detected_faces

def Landmark(img, face_rect):
    rect = dlib.rectangle(face_rect[0], face_rect[1], face_rect[0] +face_rect[2], face_rect[1] + face_rect[3])
    alignedFace = face_aligner.align(96, img, rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    return alignedFace

def FaceRecognition(alginedFace):
    # Neural Network
    net = openface.TorchNeuralNet("./nn4.small2.v1.t7")
    feature = net.forward(alginedFace)
    feature = feature.reshape(1, -1)
    # SVM Prediction
    pred = clf.predict(feature)
    person = le.inverse_transform(pred)
    return person[0]

if __name__ == "__main__":

    sc = SparkContext('local')
    sqlContext = SQLContext(sc)

    movie_path = "test.mp4"

    cap = cv2.VideoCapture(movie_path)

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = imutils.resize(frame, width = 800)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Face Detection
        # rdd_bbx = sc.parallelize(FaceDetection(gray))
        # # print(rdd_bbx.take(5))
        # bbx_list = rdd_bbx.collect()
        # for bbx in bbx_list:
        #     cv2.rectangle(frame, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]), (255, 0, 0), 2)

        rdd_bbx = sc.parallelize(FaceDetection1(gray))
        # print(rdd_bbx.take(5))
        bbx_list = rdd_bbx.collect()
        for bbx in bbx_list:
            cv2.rectangle(frame, (bbx.left(), bbx.top()), (bbx.right(), bbx.bottom()),
                          (0, 255, 0), 2)

        # # Face Alignment
        # rdd_landmark = rdd_bbx.map(lambda x: Landmark(img, x))
        # # print(rdd_landmark.take(5))
        #
        # # Face Recognition
        # rdd_label = rdd_landmark.map(lambda x: FaceRecognition(x))
        # # print(rdd_label.take(5))
        #
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # # Load Image
    # image_path = "./test.jpg"
    # img = io.imread(image_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #
    # # Face Detection
    # rdd_bbx = sc.parallelize(FaceDetection(gray))
    # # print(rdd_bbx.take(5))
    # bbx_list = rdd_bbx.collect()
    # print(bbx_list)
    # for bbx in bbx_list:
    #     cv2.rectangle(gray, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]), (255,0,0), 2)
    #
    # cv2.imshow('frame',gray)
    # cv2.waitKey(5000)
    #
    # # Face Alignment
    # rdd_landmark = rdd_bbx.map(lambda x: Landmark(img, x))
    # # print(rdd_landmark.take(5))
    #
    # # Face Recognition
    # rdd_label = rdd_landmark.map(lambda x: FaceRecognition(x))
    # # print(rdd_label.take(5))
