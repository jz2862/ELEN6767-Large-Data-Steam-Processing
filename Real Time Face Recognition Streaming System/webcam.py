import numpy as np
import cv2
import dlib
import openface
import pickle
from imutils.video import VideoStream
import imutils
import time

predictor_model = "shape_predictor_68_face_landmarks.dat"
face_aligner = openface.AlignDlib(predictor_model)
net = openface.TorchNeuralNet("nn4.small2.v1.t7")
with open('classifier.pkl', 'rb') as f:
    (le, clf) = pickle.load(f)
face_detector = dlib.get_frontal_face_detector()

vs = VideoStream(usePiCamera=0).start()
time.sleep(2.0)

while(True):
    # Capture frame-by-frame
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_faces = face_detector(gray, 1)

    for i, face_rect in enumerate(detected_faces):
        # alignedFace = face_aligner.align(96, frame, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        alignedFace = cv2.resize(frame[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()], (96, 96))
        feature = net.forward(alignedFace)
        feature = feature.reshape(1, -1)
        pred = clf.predict(feature)
        person = le.inverse_transform(pred)
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0),2)
        cv2.putText(frame, person[0], (face_rect.left(), face_rect.top() - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                    (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()
vs.stop()
