import os
import pandas
import codecs
import glob
import pandas as pd
import numpy as np
import cv2
import dlib
import sys


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


def detection_bounding1(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale(image)
    # detected_faces = face_detector(image, 1)
    boundings = []
    for k, box in enumerate(detected_faces):
        boundings.append({'left':box[0],'top':box[1],'width':box[2],'height':box[3]})
    return boundings


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


def list_to_dict(num,box):
    return {'num':num,'left':int(box[0]),'top':int(box[1]),'width':int(box[2]),'height':int(box[3])}


def write_to_file(content):
    outfile = open('accuracy.txt', 'a')
    outfile.write(content)
    outfile.write('\n')
    outfile.close()


if __name__ == '__main__':
    with open('easy.txt','r') as f:
        easyname = f.readlines()
        easyname = [i.strip() for i in easyname]
        easyname = [('_').join(i.split(' ')) for i in easyname]
    with open('recognition_data.txt','r') as f:
        flist = f.readlines()
        flist = [i.strip() for i in flist]
        flist = [[len(i.split(' '))/4]+['./WIDER_val/images/'+i.split(' ')[0]+'.jpg']+i.split(' ')[1:] for i in flist]
        new_flist = []
        for text in flist:
            new_flist.append(text[0:2]+[list_to_dict(j,text[j*4+2:j*4+6]) for j in range(text[0])])
        # print new_flist
        allresult = []
        # step = 0
        step = 0
        noteasy = True
        for i in new_flist:
            for name in easyname:
                if name in i[1]:
                    noteasy = False
            if noteasy:
                continue
            if i[2]['width'] < 100 or i[2]['height'] < 100:
                continue
            print 'haha'
            # step+=1
            # if step > 100:
            #     break
            im = cv2.imread(i[1], cv2.IMREAD_GRAYSCALE)
            box_pred = detection_bounding1(im)
            result = []


            for j in range(i[0]):
                # print j,'th for',i[0:2]
                box_label = i[j+2]
                # print 'pred',box_label
                # print 'label',box_label
                # for box_a in box_pred:
                if box_pred:
                    result.append(1.0 if max([intersection_over_union(box_a,box_label) for box_a in box_pred])>0.1 else 0.0)
                else:
                    result.append(0.0)
            ave = reduce(lambda x, y: x + y, result) / len(result)
            allresult.append(ave)
            write_to_file(' '.join(str(e) for e in i[0:2]))
            write_to_file(str(ave))
            write_to_file(str(reduce(lambda x, y: x + y, allresult) / len(allresult)))
            # step += 1
            # if step == 100:
            #     break
        write_to_file(' '.join(['accuracy over all =', str(reduce(lambda x, y: x + y, allresult) / len(allresult))]))
                    # for box_a in box_pred:
                    #     if intersection_over_union(box_a, box_label) > 0.4 and intersection_over_union(box_a, box_label) < 0.5:
                    #         cv2.rectangle(im,(box_a['left'],box_a['top']),(box_a['left']+box_a['width'],box_a['top']+box_a['height']),(0,255,0),2)
                    #         cv2.imshow('im',im)
                    #         cv2.waitKey(100)
                    #         cv2.destroyAllWindows()

                    # cv2.rectangle(im,(box_label['left'],box_label['top']),(box_label['left']+box_label['width'],box_label['top']+box_label['height']),(255,0,0),2)
            # cv2.imshow('im',im)
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()
                #     print intersection_over_union(box_a,box_label)
