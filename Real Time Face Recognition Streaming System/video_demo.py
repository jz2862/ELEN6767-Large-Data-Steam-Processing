import numpy as np
import cv2
import dlib
import openface
import pickle

from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time

print("Loading models...")
predictor_model = "shape_predictor_68_face_landmarks.dat"
face_aligner = openface.AlignDlib(predictor_model)
net = openface.TorchNeuralNet("nn4.small2.v1.t7")
with open('classifier.pkl', 'rb') as f:
    (le, clf) = pickle.load(f)
face_detector = dlib.get_frontal_face_detector()

print("Reading Movie..")
# movie_path = "./clip1/IronMan.mp4"
movie_path = "./Movies/Obama.webm"
# movie_path = "./clip4/Nicolas_Cage.mkv"

cap = cv2.VideoCapture(movie_path)

num = 1
num_faces = 0
# preframe = np.zeros((720, 1280, 3))
start = time.time()
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = imutils.resize(frame, width=800)
    if frame is None:
        continue
    # if type(frame) == 'None':
    #     break
    if num == 1 or num_faces == 0 or ((frame.astype('float') - preframe.astype('float')) ** 2).mean() > 700:
        # print('1')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector(gray, 0)
        for i, face_rect in enumerate(detected_faces):
            num_faces = i + 1
            alignedFace = face_aligner.align(96, frame, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            feature = net.forward(alignedFace)
            feature = feature.reshape(1, -1)
            pred = clf.predict(feature)
            prob = clf.predict_proba(feature)
            if max(prob[0]) < 0.1:
                person = 'Unknown'
            else:
                person = le.inverse_transform(pred)[0]
            preframe = frame
    if num_faces > 0:
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0),2)
        cv2.putText(frame, person, (face_rect.left(), face_rect.top() - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                (0, 255, 0), 1)

    num = num+1
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
end = time.time() - start
print end
cv2.destroyAllWindows()
