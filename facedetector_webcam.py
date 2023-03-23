import cv2
from random import randrange

trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#put a video file or 0 captures from webcam
webcam = cv2.VideoCapture(0)

#iterate over all frames in the video
while True:
    #read the current frame
    successful_frame_read, frame = webcam.read()
    grayscaled_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect faces and return objects as rectangle
    face_coordinates= trained_face_data.detectMultiScale(grayscaled_image)
     
    for i in face_coordinates:
       ( x,y,w,h)=i
        #draw rectangles(x,y,w,h)
       cv2.rectangle(frame, (x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),3)
    cv2.imshow('AI face detector',frame)
       #waits for 1 ms to press q key
    key=cv2.waitKey(1)
    if key==81 or key==113:
     break;
   
webcam.release()
# cv2.imshow('AI face detector',img)

# cv2.waitKey()
