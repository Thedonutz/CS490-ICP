import os
from operator import add
from pyspark import SparkContext

os.environ["SPARK_HOME"] = "C:\\Users\\plfoley\\spark-2.3.1-bin-hadoop2.7"
os.environ["HADOOP_HOME"]="C:\\Users\\plfoley\\winutils"

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    lines = sc.textFile("C:\\Users\\plfoley\\PycharmProjects\\SparkTest\\Sample.txt", 1)
    simplify = lines.flatMap(lambda x: x.split(' ')) \
        .filter(lambda x: "world" in x) \
        .distinct()

    filter2 = lines.flatMap(lambda x: x.split(' ')) \
        .filter(lambda x: "hello" in x)
    filter2.saveAsTextFile("output1")

    simplify.saveAsTextFile("output2")
    sc.stop()

