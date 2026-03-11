import RPi.GPIO as io
import paho.mqtt.client as mqtt
import Adafruit_PCA9685
pwm = Adafruit_PCA9685.PCA9685()

io.setwarnings(False)
io.setmode(io.BCM)

pwm.set_pwm_freq(60)

Motor1_PWM = 4
Motor2_PWM = 5

servo_min = 40  # Min pulse length out of 4096
servo_max = 500  # Max pulse length out of 4096
servo_mid = 400

move_speed = 3500  # Max pulse length out of 4096

io.setup(17,io.OUT)#IN1
io.setup(18,io.OUT)#IN2
io.setup(27,io.OUT)#IN3
io.setup(22,io.OUT)#IN4


def changespeed(speed):
	pwm.set_pwm(Motor1_PWM, 0, speed)
	pwm.set_pwm(Motor2_PWM, 0, speed)

def forward():
    pwm.set_pwm(0, 0, servo_mid)
    io.output(17,io.HIGH)
    io.output(18,io.LOW)
    io.output(27,io.HIGH)
    io.output(22,io.LOW)
    changespeed(move_speed)
    

def backward():
    pwm.set_pwm(0, 0, servo_mid)
    io.output(17,io.LOW)
    io.output(18,io.HIGH)
    io.output(27,io.LOW)
    io.output(22,io.HIGH)
    changespeed(move_speed)
    

def stopfcn():
    io.output(17,io.LOW)
    io.output(18,io.LOW)
    io.output(27,io.LOW)
    io.output(22,io.LOW)
    changespeed(0)
    
#this function moves the robot right
def right():
    io.output(17,io.HIGH)
    io.output(18,io.LOW)
    io.output(27,io.LOW)
    io.output(22,io.LOW)
    pwm.set_pwm(0, 0, servo_max)
    changespeed(move_speed)
    
#this function moves the robot left
def left():
    io.output(17,io.HIGH)
    io.output(18,io.LOW)
    io.output(27,io.HIGH)
    io.output(22,io.LOW)
    pwm.set_pwm(0, 0, servo_min)
    changespeed(move_speed)


MQTT_SERVER = "localhost" 
MQTT_PATH = "test_channel"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    client.subscribe(MQTT_PATH)

def on_message(client, userdata, message):
    #print("Received message '" + str(message.payload) + "' on topic '" + message.topic)
    if message.payload == b'5':
        forward()
    elif message.payload == b'0':
        stopfcn()
    elif message.payload == b'4':
        backward()
    elif message.payload == b'2':
        right()
    elif message.payload == b'1':
        left()
        
    else:
        stopfcn()

def main():
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    mqtt_client.connect(MQTT_SERVER, 1883, 60) 
    # Connect to the MQTT server and process messages in a background thread. 
    mqtt_client.loop_start() 

if __name__ == '__main__':
    print('MQTT to InfluxDB bridge')
    main()
