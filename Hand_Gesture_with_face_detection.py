import cv2
import numpy as np
from os.path import isfile, join
from os import listdir

import mediapipe as mp
import paho.mqtt.publish as publish

data_path = 'C:/Users/Lenovo/PycharmProjects/OpenCVpython/Tutorial 3/Recordings/'
only_files = [f for f in listdir(data_path) if isfile(join(data_path,f))]

MQTT_SERVER = "172.24.1.1"
MQTT_PATH = "test_channel"

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

tipIds = [4, 8, 12, 16, 20]

Training_Data, Labels = [], []

for i, files in enumerate(only_files):
    img_path = data_path + only_files[i]
    images = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Complete")

face_classifier = cv2.CascadeClassifier('C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(imgGray, 1.3, 5)

    if faces is ():
        return img, []

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+h, y+w), (0,255,255), 2)
        roi = img[x:x+h, y:y+w]
        #roi = cv2.resize(roi, (200,200))
    return img, roi

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hand.Hands(min_detection_confidence=0.5,
                   min_tracking_confidence=0.5, max_num_hands=1) as hands:

    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)
            # print("result is", result)
            if result[1] < 500:
                match = int(100*(1-(result[1])/300))
                # print("Match is:", match)
                if match > 82:
                    cv2.putText(image, "Me", (250,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                else:
                    cv2.putText(image, "Other", (250,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except:
            cv2.putText(image, "No User Found", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_classifier.detectMultiScale(image)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lmList = []

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                myHands = results.multi_hand_landmarks[0]
                for id, lm in enumerate(myHands.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
        fingers = []

        if len(lmList) != 0:
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            finger_counter = fingers.count(1)

            #print("Finger counter: ", finger_counter)
            publish.single(MQTT_PATH, finger_counter, hostname=MQTT_SERVER)

            if finger_counter == 5:
                cv2.putText(image, 'Forward', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

            elif finger_counter == 4:
                cv2.putText(image, 'Backward', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

            elif finger_counter == 0:
                cv2.putText(image, 'Stop', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

            elif finger_counter == 2:
                cv2.putText(image, 'Right', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

            elif finger_counter == 1:
                cv2.putText(image, 'Left', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)


        cv2.imshow("Web Cam", image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()