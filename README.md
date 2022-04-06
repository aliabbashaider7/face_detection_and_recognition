# Face Detection, Recognition and Gender Classification:
An ultra fast and accurate python implementation to detect, recognize and classify gender of the face of a person.

# Pre-requisits:
The code is tested on ubuntu 20.04 with python3.6

The code is tested with tensorflow==1.15.0 and pytorch>=1.7.1

pip3 install -r requirements.txt

Note: In case dlib fails to build, You can install it from the source from official dlib github such as on jetson toolkits.

# Face Recognition Training:
Dataset sample has been given in faces_database directory. Make sure to keep the folders that way and the names inside of the folder like given in sample dataset. After preparing dataset accordingly, run training.py script which outputs database.npy file inside of vitals directory that will be used in inference section to recognize trained faces.

# Inference
Run inference.py and give path to your test image. Output images will be saved inside of results directory.
