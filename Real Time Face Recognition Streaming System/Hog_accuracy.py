import os
import pandas
import codecs
import glob
import pandas as pd
import numpy as np
import cv2
import dlib
import sys



def read_video(path, frame_number):
    cap = cv2.VideoCapture(path)
    cap.set(1, frame_number-1)
    res, frame = cap.read()
    # while(cap.isOpened()):
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame
    # cv2.imshow('frame', frame)
    # while True:
    #     ch = 0xFF & cv2.waitKey(1)
    #     if ch == 5:
    #         break
    # cap.release()


def detection_bounding(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detector = dlib.get_frontal_face_detector()
    detected_faces = face_detector(image, 1)
    boundings = []
    for k, box in enumerate(detected_faces):
        width = box.right()-box.left()
        height = box.bottom()-box.top()
        boundings.append({'left':box.left(),'top':box.top(),'width':width,'height':height})
    return boundings


def read_track_files(path):
    filename_list=os.listdir(path)
    h=[]
    for filename in filename_list:
        if filename.endswith('txt'):
            h.append(filename)
    return h


def read_bounding(path):
    f = open(path, 'r')
    x = f.readlines()
    f.close()
    y = [[int(j) for j in i.split(',')] for i in x]
    track_box = []
    for i in y:
        track_box.append({'frame_number':i[0],'left':i[1],'top':i[2],'width':i[3],'height':i[4]})
    return track_box


def intersection_over_union(box_a, box_b):
    # Determine the coordinates of each of the two boxes
    #box_a = [a['left'],a['top'],a['left']+a['width'],a['top']+a['height']]
    xA = max(box_a['left'], box_b['left'])
    yA = max(box_a['top'], box_b['top'])
    xB = min(box_a['left']+box_a['width'], box_b['left']+box_b['width'])
    yB = min(box_a['top']+box_a['height'], box_b['top']+box_b['height'])
    if xB < xA or yB < yA:
        return 0.0
    # Calculate the area of the intersection area
    area_of_intersection = (xB - xA + 1) * (yB - yA + 1)

    # Calculate the area of both rectangles
    box_a_area = (box_a['width'] + 1) * (box_a['height'] + 1)
    box_b_area = (box_b['width'] + 1) * (box_b['height'] + 1)
    # Calculate the area of intersection divided by the area of union
    # Area of union = sum both areas less the area of intersection
    iou = area_of_intersection / float(box_a_area + box_b_area - area_of_intersection)
    # Return the score
    return iou


def iter_movie(filename):
    allarea = []
    step = 0
    for j in read_track_files('./tracks_final/'+filename+'/'):
        bounding_result = read_bounding('./tracks_final/'+filename+'/'+j)
        frame = read_video('./movies/'+i, int(bounding_result[0]['frame_number']))
        bounding_prediction = detection_bounding(frame)
        if bounding_prediction:
            area = max([intersection_over_union(bounding_prediction[gua],bounding_result[0]) for gua in range(len(bounding_prediction))])
            allarea.append(area)
        print step,reduce(lambda x, y: x + y, allarea) / len(allarea)
        step += 1
    return reduce(lambda x, y: x + y, allarea) / len(allarea)


def changedir(path='./tracks_final'):
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)) is True:
            if len(file) > 11:
                newname = file[0:11]
                os.rename(os.path.join(path,file),os.path.join(path,newname))


if __name__ == '__main__':
    with open('vidNames.txt') as f:
        flist = f.readlines()
        flist = [i.strip() for i in flist]
    for i in os.listdir('./movies/'):
        for fn in range(len(flist)):
            if os.path.isfile(os.path.join('./movies/', i)) and flist[fn] in i:
                print './movies/'+i
                accuracy = iter_movie(flist[fn])
                orig_stdout = sys.stdout
                f = open('out'+'_'+flist[fn]+'.txt', 'w')
                sys.stdout = f
                print './movies/'+i+accuracy
                sys.stdout = orig_stdout
                f.close()
                # re.sub('"(.*?)"', r'\1', s)
