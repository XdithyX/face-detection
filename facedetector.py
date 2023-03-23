import cv2
# import sys
# print(sys.path)

from random import randrange

#load some pre trained datanon face frontals from opencv(haar cascade algorithm)
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#read image
img=cv2.imread('testdata/leo.jpg')

#image grayscaling
grayscaled_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect faces and return objects as rectangle
face_coordinates= trained_face_data.detectMultiScale(grayscaled_image)

for i in face_coordinates:
 ( x,y,w,h)=i
#draw rectangles(x,y,w,h)
 cv2.rectangle(img, (x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

#show image
cv2.imshow('AI face detector',img)

#pause execution of program
cv2.waitKey()

print("code completed")