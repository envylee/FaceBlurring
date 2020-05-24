# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CASCADE_SCALE_IMAGE
	)
	#global out
	#print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		mask = np.zeros(np.shape(frame), dtype=np.uint8)
		mask = cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
		blurred = cv2.GaussianBlur(frame, (51,51), 11)
		frame = np.where(mask==np.array([255, 255, 255]), blurred, frame)

	# Display the resulting frame
	cv2.imshow('frame', frame)

	# press q to exit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
