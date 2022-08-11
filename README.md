# Gesture-Control-Robot
The raspberry pi based robot moves as per user hand gestures.
When user shows one finger, the robot rotates in left direction
When user shows two finger, the robot rotates in right direction
When user shows four finger, the robot moves in reverse direction
When user shows five finger, the robot moves in forward direction
When user shows fist, the robot stops.

For hand detection and tracking, MediaPipe library is used.
Find the link here https://google.github.io/mediapipe/solutions/hands.html

#Face Detection Model
Step 1: Get the user images. Collected around 800 samples
Step 2: Train them using the haarcascade classifiers
Step 3: Predict the accuracy and put the name of user.

#Libraries Used
1. In Pycharm:

pip install opencv-python

pip install mediapipe

pip install paho-mqtt

2. In Raspberry Pi

sudo pip install paho-mqtt

sudo apt-get install -y mosquitto mosquitto-clients

sudo systemctl enable mosquitto
