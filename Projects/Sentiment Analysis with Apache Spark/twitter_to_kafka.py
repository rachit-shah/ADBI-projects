import json
import time
from kafka import SimpleProducer, KafkaClient
import configparser

# Note: Some of the imports are external python libraries. They are installed on the current machine.
# If you are running multinode cluster, you have to make sure that these libraries
# and currect version of Python is installed on all the worker nodes.

class TweeterStreamProducer():
    """ A class to read the tweet stream and push it to Kafka"""

    def __init__(self):
        client = KafkaClient("localhost:9092")
        self.producer = SimpleProducer(client, async = True,
                          batch_send_every_n = 1000,
                          batch_send_every_t = 10)

    def on_status(self, status):
        """ This method is called whenever new data arrives from live stream.
        We asynchronously push this data to kafka queue"""
        msg =  status
        #print(msg)
        try:
            self.producer.send_messages('twitterstream', msg)
        except Exception as e:
            print(e)
            return False
        return True

    def on_error(self, status_code):
        print("Error received in kafka producer")
        return True # Don't kill the stream

    def on_timeout(self):
        return True # Don't kill the stream


if __name__ == '__main__':
    # To simulate twitter stream, we will load tweets from a file in a streaming fashion
    f = open('16M.txt')
    stream = TweeterStreamProducer()
    i=0
    for data in f:
        stream.on_status(data.strip())
        i+=1
        if i % 10000 == 0:
            print ("Pushed ", i, " messages")