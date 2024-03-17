# vashisht-hackathon
<h1 align="center" style="border-bottom: none">
    <b>
        <a href="https://www.google.com"> Driver Drowsiness Detection and Alerting System </a><br>
    </b>
    ⭐️A Buzzer to live⭐️ <br>
</h1>

[`Demo video link `](![drowsiness_detector_demo](https://github.com/Harinithiruveedula05/vashisht-hackathon/assets/152847148/525c0cd4-a6af-4219-a634-64a35df36c0e)
)
Driver Drowsiness detection and alerting system detects if a driver is drowsy or not using their eye moments and alerts them. This Script detects if a person is drowsy or not,using dlib and eye aspect ratio calculations. Uses webcam video feed as input.
## Team Details
`Team number` : VH241

| Name    | Email           |
|---------|-----------------|
| T.Harini Sai | harinithiruveedula05@gmail.com |
| K.Guru Prasad Reddy | 9921004353@klu.ac.in|
| v .pravallika | 99210042087@klu.ac.in |
|D.Abeed Salman|9921004186@klu.ac.in|

![istockphoto-1248728931-612x612](https://github.com/Harinithiruveedula05/vashisht-hackathon/assets/152847148/333a012b-3823-4945-a453-8ee14a11929c)
![web_cam_face_detection](https://github.com/Harinithiruveedula05/vashisht-hackathon/assets/152847148/5e23db72-f098-4b51-aaa3-e0f03b814963)
![drowsiness_detection](https://github.com/Harinithiruveedula05/vashisht-hackathon/assets/152847148/4028a45c-4f92-400a-bfb3-e86b61c126c1)

## Problem statement 
The problem of drowsy driving remains a significant concern worldwide, leading to a high number of accidents, injuries, and fatalities on the road. The objective of this project is to develop a Driver Drowsiness and Alerting System using deep learning techniques to mitigate the risks associated with drowsy driving!

## About the project
The majority of today's traffic accidents are caused by driver errors and carelessness. Drowsiness, intoxication, and reckless driving are the leading causes of major driver errors. 
This project focuses on a driver drowsiness detection system for the Intelligent Transportation System, which focuses on anomalous behavior displayed by the driver when using a computer. 
**HERE** 
The program contains 3 files, which are
## Files
**face_and_eye_detector_single_image.py** -Detects face and eye from a single image.
**face_and_eye_detector_webcam_video.py** -Detects face and eye in a webcam feed by user.
**drowsiness_detect.py** - Shows demo ( It detects the driver drowsiness and alert them by producing an alert based on our threshold value 
## Technical implemntaion 
![Screenshot (276)](https://github.com/Harinithiruveedula05/vashisht-hackathon/assets/152847148/dbc4bd22-cc3a-4b7c-97ba-4fe9f9ecef07)

Download `shape_predictor_68_face_landmarks.dat.bz2` from [Shape Predictor 68 features](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)                                                       Extract the file in the project folder using ``bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2``
**REQUIREMENTS**
numpy==1.15.2
dlib==19.16.0
pygame==1.9.4
imutils==0.5.1
opencv_python==3.4.3.18
scipy==1.1.0
Use `pip install -r requirements.txt`to install the given requirements.
## Usage
### Detect Face and Eyes in a Single Image
Put your file to be detected in **images** folder with name **test.jpeg** or change the file path in Line : 9 face_and_eye_detector_single_image.py` to your image file. 
Run script using:

    python face_and_eye_detector_single_image.py
### Detect Face and Eyes in a Webcam Feed
Run script using:

    python face_and_eye_detector_webcam_video.py
### Drowsiness Detection
Run script using:

    python drowsiness_detect.py
    The algorithm for Eye Aspect Ratio was taken from pyimagesearch.com blog, by Adrian RoseBrock.
## ALGORITHM
We use dlib for face detection and facial landmarks along with the eye aspect ratio calculation.
1) we use open cv for importing the required libraries
2) we uses the Haar Cascade classifier for face detection
3) The dlib face detector and shape predictor are loaded using the provided files
4) In a main loop of code Each frame is read from the webcam and flipped.
5)Facial points are detected using both the Haar Cascade classifier and the dlib face detector.
6)Rectangles are drawn around detected faces.
7)For each detected face:
8)Facial landmarks are extracted using the shape predictor.Eye aspect ratio is calculated for both eyes.Convex hulls are used to draw contours around the eyes.If the average eye aspect ratio is below the threshold, the drowsiness counter is incremented.If the counter exceeds the specified consecutive frames threshold, an alarm sound is played, and a warning message is displayed on the frame.If the eye aspect ratio is above the threshold, the counter is reset, and the alarm sound
is stopped.
9)The video feed is displayed, and if the 'q' key is pressed, the loop is exited.
This algorithm continuously monitors the eye aspect ratio to determine if a person is drowsy, triggering an alarm when necessary. The dlib library is used for accurate facial landmark detection, and OpenCV is used for video capture and visualization.
## EXECUTION
## Image for detecting eyes and face
![web_cam_face_detection](https://github.com/Harinithiruveedula05/Driver-Drowsiness-detection-and-alerting-system/assets/152847148/4f9892c8-4121-4c20-a71b-71c9bbf06194)

## Image for detecting drowsiness

![drowsiness_detection](https://github.com/Harinithiruveedula05/Driver-Drowsiness-detection-and-alerting-system/assets/152847148/e7eadf57-5825-4bd6-8b2d-6343945b2213)


## CODE
#Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import time
import dlib
import cv2

#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

#Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 0.3

#Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

#COunts no. of consecutuve frames below threshold value
COUNTER = 0

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

#Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#Start webcam video capture
video_capture = cv2.VideoCapture(0)

#Give some time for camera to initialize(not required)
time.sleep(2)

while(True):
    #Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #Detect facial points
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Detect if eye aspect ratio is less than threshold
        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            pygame.mixer.music.stop()
            COUNTER = 0

    #Show video feed
    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()


## Declaration
We confirm that the project showcased here was either developed entirely during the hackathon or underwent significant updates within the hackathon timeframe. We understand that if any plagiarism from online sources is detected, our project will be disqualified, and our participation in the hackathon will be revoked.



