Asynchronous Face Blurring

To use the program, 
1. run python3 setup.py to setup the database using sqlite3

2. run python3 recordface_webcam.py (video_path) to obtain the face for the training for all faces, file will stored in dataset directory

3. run python3 trainer.py to train the LBPH face recognizer

4. run python3 detector_webcam.py (video_path) to do async blurring

Thanks for your attention!
