# Gesture-Control-Robot

## Project Overview
This project implements a **gesture-controlled robot** using a **Raspberry Pi** and **hand tracking with MediaPipe**.  
The robot responds to specific hand gestures and moves accordingly, making it an interactive demonstration of computer vision and robotics integration.  

https://github.com/user-attachments/assets/0ca84fa8-69bb-4c16-9d66-594d3e455bd5

---

## Gesture Control Mapping
The robot movement is mapped to the number of fingers detected:
- **1 Finger** → Rotate **Left**  
- **2 Fingers** → Rotate **Right**  
- **4 Fingers** → Move **Backward**  
- **5 Fingers** → Move **Forward**  
- **Fist (0 Fingers)** → **Stop**  

Hand detection and tracking is implemented using [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html), which provides efficient real-time hand landmark recognition.

---

## Communication with MQTT
The robot uses the **MQTT protocol** for communication between the gesture detection system (running on a PC) and the Raspberry Pi:  
- **Publisher**: The gesture recognition code publishes commands (e.g., "forward", "left") to an MQTT topic.  
- **Broker**: Mosquitto MQTT broker manages message delivery.  
- **Subscriber**: The Raspberry Pi rover subscribes to the topic and executes corresponding movement commands.  

This design enables a lightweight, real-time, and scalable way of sending commands, making the system more modular and network-friendly.

---

## Face Detection Model (Additional Feature)
A face recognition system is also implemented using **Haarcascade Classifiers**:
1. Collect ~800 face images per user.  
2. Train the model using Haarcascade.  
3. Predict user identity during runtime and display the name with accuracy.  

---

## Libraries & Tools Used

### On PC / Development Machine
pip install opencv-python
pip install mediapipe
pip install paho-mqtt

### On Raspberry Pi

sudo pip install paho-mqtt
sudo apt-get install -y mosquitto mosquitto-clients
sudo systemctl enable mosquitto
