from pyspark import SparkContext
import os

os.environ["SPARK_HOME"] = "/Users/jm051781/Documents/spark-2.3.1-bin-hadoop2.7/"

sc = SparkContext()

sortMe = sc.parallelize([3, 6, 2, 1, 4])

sorted = sortMe.map(lambda x: (x, 1))\
    .reduceByKey(lambda a, b: a + b) \
    .sortByKey(lambda tup: len(tup[1])) \
    .collect()

saveList = []
for x in range (len(sorted)):
    saveList.append(sorted[x][0])

print(saveList)


