import cv2
vidcap = cv2.VideoCapture('VfE_html5.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
