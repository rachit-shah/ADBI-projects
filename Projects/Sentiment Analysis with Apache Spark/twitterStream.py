from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def main():
	conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
	sc = SparkContext(conf=conf)
	sc.setLogLevel("WARN")
	ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
	ssc.checkpoint("checkpoint")

	pwords = load_wordlist("positive.txt")
	nwords = load_wordlist("negative.txt")

	counts = stream(ssc, pwords, nwords, 100)
	#print("counts : ",counts)
	make_plot(counts)


def make_plot(counts):
	"""
	Plot the counts for the positive and negative words for each timestep.
	Use plt.show() so that the plot will popup.
	"""
	fig = plt.figure()
	pcounts = [x[0][1] for x in counts]
	ncounts = [x[1][1] for x in counts]
	timestamp = range(0,len(counts))
	plt.plot(timestamp,pcounts,"bo-",label='positive')
	plt.plot(timestamp,ncounts,"go-",label='negative')
	plt.legend(loc='upper left')
	plt.ylabel('Word count')
	plt.xlabel('Time step')
	fig.savefig("plot.png")


def load_wordlist(filename):
	""" 
	This function should return a list or set of words from the given filename.
	"""
	return set(open(filename).read().splitlines())


def updateFunction(newValues, runningCount):
	if runningCount is None:
		runningCount = 0
	return sum(newValues, runningCount)  # add the new values with the previous running count to get the new 

def findCount(x,pnlist):
	return len(pnlist.intersection(set(x.lower().split(" "))))
	
def stream(ssc, pwords, nwords, duration):
	kstream = KafkaUtils.createDirectStream(
		ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
	tweets = kstream.map(lambda x: x[1])

	# Each element of tweets will be the text of a tweet.
	# You need to find the count of all the positive and negative words in these tweets.
	# Keep track of a running total counts and print this at every time step (use the pprint function).
	pairs = tweets.flatMap(lambda x: [("positive",findCount(x,pwords)),("negative",findCount(x,nwords))])
	wordCounts = pairs.reduceByKey(lambda x, y: x + y)
	runningCounts = pairs.updateStateByKey(updateFunction)
	runningCounts.pprint()


	# Let the counts variable hold the word counts for all time steps
	# You will need to use the foreachRDD function.
	# For our implementation, counts looked like:
	#   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
	counts = []
	wordCounts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))

	ssc.start()                         # Start the computation
	ssc.awaitTerminationOrTimeout(duration)
	ssc.stop(stopGraceFully=True)

	return counts


if __name__=="__main__":
	main()
