import cv2
import os
import sqlite3
import time
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils

# Connect SQLite3 database
conn = sqlite3.connect('database.db')
db = conn.cursor()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")
ap.add_argument("-b", "--blur", help="blur face with flag", action="store_true")
ap.add_argument("-d", "--debug", help="debug mode", action="store_true")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# Assign the training data file
fname = "recognizer/trainingData.yml"
if not os.path.isfile(fname):
    print("Please train the data first")
    exit(0)

lastDetectedAt = 0
detectInterval = 5 # 1/n second, for reducing overhead
lastUnlockedAt = 0
unlockDuration = 5 # n second

# Font used for display
font = cv2.FONT_HERSHEY_SIMPLEX

# Connect to video source
#vSource = "rtsp://192.168.1.100:8554/live.sdp" # RTSP URL of IP Cam
vSource = 0
if args.get("video", False):
    vSource = args["video"]
vStream = cv2.VideoCapture(vSource)

# initialize the FPS throughput estimator
fps = None

# Setup Classifier for detecting face
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Setup LBPH recognizer for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Load training data
recognizer.read(fname) # change to read() for LBPHFaceRecognizer_create()

while vStream.isOpened():
    # initialize the location of faces
    faces_location = {}

    ret, frame = vStream.read() # Read frame
    if not ret:
        break

    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    #detect the exsistance of face and locate in time duration
    timeElapsed = time.time() - lastDetectedAt
    if timeElapsed > 1./detectInterval:
        lastDetectedAt = time.time()

        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert captured frame to grayscale
        faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5) # Detect face(s) inside the frame

        for (x, y, w, h) in faces:
            # Try to recognize the face using recognizer
            roiGray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roiGray)
            print(id_, conf)

            # If recognized face has enough confident (<= 70),
            # retrieve the user name from database,
            # draw a rectangle around the face,
            # print the name of the user
            if conf <= 70:
                faces_location[id_] = (x, y, w, h)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # retrieve user name from database
                db.execute("SELECT `name` FROM `users` WHERE `id` = (?);", (id_,))
                result = db.fetchall()
                name = result[0][0]

                # User is detected and blur is applied to the rectangle
                #mask = np.zeros(np.shape(frame), dtype=np.uint8)
                #mask = cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
                #blurred = cv2.GaussianBlur(frame, (51,51), 11)
                #frame = np.where(mask==np.array([255, 255, 255]), blurred, frame)

                lastUnlockedAt = time.time()
                print("[Predict] " + str(id_) + ":" + name + " (" + str(conf) + ")")
                cv2.putText(frame, name, (x+2,y+h-5), font, 1, (150,255,0), 2)
            else:
                #confident level is not high enough
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                print("[Fail] " + str(conf))

    #KCF object tracking update by faces_location list
    if not (len(faces_location) == 0):
        for name in faces_location:
            initBB = faces_location[name]
            print(initBB)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            tracker.init(frame, initBB)
            fps = FPS().start()

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # with blur flag enable
        if args.get("blur", False):
            mask = np.zeros(np.shape(frame), dtype=np.uint8)
            mask = cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
            blurred = cv2.GaussianBlur(frame, (51,51), 11)
            frame = np.where(mask==np.array([255, 255, 255]), blurred, frame)

        # update the FPS counter
        fps.update()
        fps.stop()

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    cv2.imshow("Face Recognizer", frame)

    # Press ESC or 'q' to quit the program
    key = cv2.waitKey(1) & 0xff
    if key == 27 or key == ord('q'):
        break

# Clean up
vStream.release()
conn.close()
cv2.destroyAllWindows()
