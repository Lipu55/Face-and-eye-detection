# Face-and-eye-detection
How to detect face and eye from live video
#Face & Eye Detection using HAAR Cascade Classifiers
import numpy as np
import cv2
# We point OpenCV's CascadeClassifier function to where our 
# classifier (XML file format) is stored
face_classifier=cv2.CascadeClassifier(r"D:\Downloads\haarcascade_frontalface_default.xml")
# Load our image then convert it to grayscale
image=cv2.imread(r"C:\Users\MRUTYUNJAY\Pictures\Camera Roll\WIN_20230512_09_00_22_Pro.jpg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# Our classifier returns the ROI of the detected face as a tuple
# It stores the top left coordinate and the bottom right coordiantes
faces=face_classifier.detectMultiScale(gray,1.3,5)
# When no faces detected, face_classifier returns and empty tuple
if len(faces) == 0:
    print("No faces found")
# We iterate through our faces array and draw a rectangle
# over each face in faces    
for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('Face Detection',image)
    cv2.waitKey(0)
cv2.destroyAllWindows()

#Let's combine face and eye detection
import numpy as np
import cv2
face_classifier=cv2.CascadeClassifier(r"D:\Downloads\haarcascade_frontalface_default.xml")
eye_classifier=cv2.CascadeClassifier(r"D:\Downloads\haarcascade_eye.xml")
img=cv2.imread(r"C:\Users\MRUTYUNJAY\Pictures\Camera Roll\WIN_20230512_09_00_22_Pro.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_classifier.detectMultiScale(gray,1.3,5)
# When no faces detected, face_classifier returns and empty tuple
if len(faces) ==0:
    print("No Face Found")
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    roi_gray=gray[y:y+h, x:x+w]
    roi_color=img[y:y+h, x:x+w]
    eyes=eye_classifier.detectMultiScale(roi_gray)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
        cv2.imshow('img',img)
        cv2.waitKey(0)
cv2.destroyAllWindows()
# Let's make a live face & eye detection, keeping the face inview at all times
import cv2
import numpy as np
face_classifier=cv2.CascadeClassifier(r"D:\Downloads\haarcascade_frontalface_default.xml")
eye_classifier=cv2.CascadeClassifier(r"D:\Downloads\haarcascade_eye.xml")
def face_detector(img,size=0.5):
     # Convert image to grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if len(faces) ==0:
        return img
    for (x,y,w,h) in faces:
        x=x-50
        w=w+50
        y=y-50
        h=h+50
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes=eye_classifier.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    roi_color=cv2.flip(roi_color,1)
    return roi_color
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    cv2.imshow('Our Face Extractor',face_detector(frame))
    if cv2.waitKey(1)==13: #13 is the Enter Key
        break
cap.release()

cv2.destroyAllWindows()        
