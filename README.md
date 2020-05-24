# Asynchronous Face Blurring





## Usage

1. run `python3 setup.py` to setup the database using sqlite3
2. run `python3 recordface.py [-v] [video_path]` to obtain the face for the training for all faces, file will stored in dataset directory
3. run `python3 trainer.py` to train the LBPH face recognizer
4. run `python3 detector.py [-v] [video_path]` to do asynchronous face recognition
5. run `python3 detector.py -b [-v] [video_path]` to do asynchronous face blurring





## Contribution

LEE, Pak Yin: face recognition

LAU, Tsz Yui: object tracking

Project fork from:

https://github.com/mickey9801/opencv_facerecognition

https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/





Thanks for your attention!
