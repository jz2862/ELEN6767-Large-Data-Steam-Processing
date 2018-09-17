import numpy as np
import cv2
import dlib
import openface
import pickle
import time

def intersection_over_union(box_a, box_b):
    # Determine the coordinates of each of the two boxes
    #box_a = [a['left'],a['top'],a['left']+a['width'],a['top']+a['height']]
    xA = max(box_a.left(), box_b.left())
    yA = max(box_a.top(), box_b.top())
    xB = min(box_a.right(), box_b.right())
    yB = min(box_a.bottom(), box_b.bottom())
    if xB < xA or yB < yA:
        return 0.0
    # Calculate the area of the intersection area
    area_of_intersection = (xB - xA + 1) * (yB - yA + 1)

    # Calculate the area of both rectangles
    box_a_area = (box_a.right()-box_a.left() + 1) * (box_a.bottom()-box_a.top() + 1)
    box_b_area = (box_b.right()-box_b.left() + 1) * (box_b.bottom()-box_b.top() + 1)
    # Calculate the area of intersection divided by the area of union
    # Area of union = sum both areas less the area of intersection
    iou = area_of_intersection / float(box_a_area + box_b_area - area_of_intersection)
    # Return the score
    return iou


print("Loading models...")
predictor_model = "shape_predictor_68_face_landmarks.dat"
face_aligner = openface.AlignDlib(predictor_model)
net = openface.TorchNeuralNet("nn4.small2.v1.t7")
with open('classifier.pkl', 'rb') as f:
    (le, clf) = pickle.load(f)

print("Reading Movie..")
# movie_path = "./clip1/IronMan.mp4"
movie_path = "./Presentation/Bechham.mp4"
# movie_path = "./clip4/Nicolas_Cage.mkv"
accuracy = []
cap = cv2.VideoCapture(movie_path)
ret = True
movie = []
ret, frame = cap.read()
while (ret):
    ret, frame = cap.read()
    if ret == False:
        continue
    frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)
    movie.append((ret,frame))
print 'finished storing'


bound = []
ret, preframe = movie[0]
num = 1
num_faces = 0
count = 1
acc = []
for ret, frame in movie[1:]:
    bbs = face_aligner.getAllFaceBoundingBoxes(frame)
    result = []
    for i, face_rect in enumerate(bbs):
        num_faces = i + 1
        alignedFace = face_aligner.align(96, frame, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        feature = net.forward(alignedFace)
        feature = feature.reshape(1, -1)
        predictions = clf.predict_proba(feature).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        if confidence < 0.15:
            person = 'Unknown'
        result.append([person, face_rect])
    for person, face_rect in result:
        bound.append(face_rect)
        acc.append(1 if person == 'David Beckham' else 0)
    preframe = frame


for threshold in xrange(0, 1000, 50):
    ret, preframe = movie[0]
    num = 0
    num_faces = 0
    count = 1
    acc = []
    s_time = time.time()
    for ret, frame in movie[1:]:
        # print 'The %dth frame' % num
        # frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)
        # if num == 896:
        #     break
        # if num > 1:
            # print ((frame.astype('float') - preframe.astype('float')) ** 2).mean()
        if num == 1 or num_faces == 0 or ((frame.astype('float') - preframe.astype('float')) ** 2).mean() > threshold:
            bbs = face_aligner.getAllFaceBoundingBoxes(frame)
            result = []
            for i, face_rect in enumerate(bbs):
                num_faces = i + 1
                alignedFace = face_aligner.align(96, frame, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                feature = net.forward(alignedFace)
                feature = feature.reshape(1, -1)
                predictions = clf.predict_proba(feature).ravel()
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = predictions[maxI]
                if confidence < 0.15:
                    person = 'Unknown'
                result.append([person, face_rect])

            for person, face_rect in result:
            #     cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0),2)
            #     cv2.putText(frame, person, (face_rect.left(), face_rect.top() - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
            #                 (0, 255, 0), 1)
                # print person
                acc.append(intersection_over_union(face_rect, bound[num]))
            preframe = frame
        else:
            for person, face_rect in result:
                # cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0),2)
                # cv2.putText(frame, person, (face_rect.left(), face_rect.top() - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                #         (0, 255, 0), 1)
                # print person
                acc.append(intersection_over_union(face_rect, bound[num]))
        # Display the resulting frame
        num = num+1
        # cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    e_time = time.time() - s_time
    accuracy.append((threshold, reduce(lambda x, y: x + y, acc) / float(len(acc)), 600/e_time))
    print 'accuracy = ', accuracy
    # When everything done, release the capture
    # e_time = time.time() - s_time
    cv2.destroyAllWindows()
    print('Number of Frames = %d' % num)
    # print('Total Running Time = %f'% e_time)
    # print('System Throughput = %f'% (num / e_time))
