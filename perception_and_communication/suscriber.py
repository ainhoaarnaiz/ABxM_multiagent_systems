import time
import paho.mqtt.client as paho

TOPIC_CONTROL = "/robot_control"
TOPIC_CAMERA = "/camera"

# Global variable to store the message payload
last_message = None

# Callbacks definitions
def on_connect(client, userdata, flags, rc, properties=None):
    print("CONNACK received with code %s." % rc)

def on_publish(client, userdata, mid, properties=None):
    print("mid: " + str(mid))

def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

def on_message(client, userdata, msg):
    global last_message
    last_message = msg.payload.decode('utf-8')
    print(msg.topic + " " + str(msg.qos) + " " + last_message)

# Create a client instance
client = paho.Client(client_id="", protocol=paho.MQTTv5)
client.on_connect = on_connect
client.on_publish = on_publish
client.on_subscribe = on_subscribe
client.on_message = on_message

# Connect to HiveMQ Cloud on port 1883 (default for MQTT over TLS)
client.connect("mqtt-dashboard.com", 1883)

# Subscribe to topic
client.subscribe(TOPIC_CAMERA, qos=0)

# Start the loop in the background
client.loop_start()

# Main loop to periodically check the last_message
try:
    while True:
        if last_message:
            print(f"Last message received: {last_message}")
        time.sleep(5)
except KeyboardInterrupt:
    print("Exiting...")
    client.loop_stop()
    client.disconnect()
