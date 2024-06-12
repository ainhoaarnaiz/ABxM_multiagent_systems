import paho.mqtt.client as paho

# MQTT Callbacks definitions
def on_connect(client, userdata, flags, rc, properties=None):
    print("CONNACK received with code %s." % rc)

def on_publish(client, userdata, mid, properties=None):
    print("mid: " + str(mid))

def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))

def on_disconnect(client, userdata, rc, properties=None):
    print("MQTT Disconnected")


def setup(client,topic,previousCheck):
    # MQTT Setup
    client.on_connect = on_connect
    client.on_publish = on_publish
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    # Last Will and Testament (LWT) and must be before connect
    client.will_set(topic, payload=previousCheck, qos=1, retain=True) 
    
    ## Connect to HiveMQ Cloud on port 1883 (default for MQTT over TLS)
    client.connect("mqtt-dashboard.com",1883)
    
    ## Send an Intial Value to the Broker)
    client.publish(topic, payload=previousCheck, qos=1, retain=True)
    print("Data published to MQTT broker - " + str(previousCheck))

def publish(value, client, topic, qos=1):
    try:
        if not client.is_connected():
            client.reconnect()
        client.publish(topic, payload=value, qos=qos, retain=True)
        print("Data published to MQTT broker - " + str(value))
    except Exception as e:
        print(f"Failed to publish message: {e}")