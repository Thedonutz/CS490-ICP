import os
from operator import add
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.functions import col
from pyspark.sql import *
#from pyspark.sql.functions import collectlist, size

os.environ["SPARK_HOME"] = "C:\\Users\\plfoley\\spark-2.3.1-bin-hadoop2.7"
os.environ["HADOOP_HOME"]="C:\\Users\\plfoley\\winutils"

spark = SparkSession \
    .builder \
    .appName("Lesson10") \
    .getOrCreate()

#Create Dataframe
df = spark.read.load("C:\\Users\\plfoley\\Desktop\\UMKC\\BigDataProgrammingSummer2018\\Lesson10\\ConsumerComplaints.csv",
                     format="csv", sep=",", inferSchema="true", header="true")

#Print the Dataframe
print("\nDataFrame: 1")
df.show()

print("\nCount the number of repeated records")
print("\n")
print((df.count())-(df.dropDuplicates().count()))

unionDF = df.unionAll(df)
unionDF.registerTempTable("Complaints")
DB = spark.sql("SELECT Company from complaints").show(200)

df.groupBy(df.columns[17]).agg(collect_list(df.columns[17]).alias("ids")).where(size("ids")>1).show()


df.groupBy("Zip Code").count().show()


df1 = df.alias('df1')
df2 = df.alias('df2')

df1.join(df2, df1.ID == df2.ID).select('df1.*').show()

print(df.rdd.take(13).pop())

df.groupBy(df.columns).count().where(f.col('count')>1).select(f.sum('count')).show()

print("\nSaved Data to File")
df.write.save("C:\\Users\\plfoley\\Desktop\\UMKC\\BigDataProgrammingSummer2018\\Lesson10\\SavedConsumerComplaints.csv", format="csv")
