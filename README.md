# Asynchronous Face Blurring





## Usage

1. run `python3 setup.py` to setup the database using sqlite3
2. run `python3 recordface_webcam.py (video_path)` to obtain the face for the training for all faces, file will stored in dataset directory
3. run `python3 trainer.py` to train the LBPH face recognizer
4. run `python3 detector_webcam.py (video_path)` to do asynchronous face recognition
5. run `python3 detector_webcam.py -b (video_path)` to do asynchronous face blurring





## Contribution

LEE, Pak Yin: face recognition

LAU, Tsz Yui: object tracking





Thanks for your attention!
