from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import shutil
# Load training data
from pyspark.python.pyspark.shell import spark
import os

os.environ["SPARK_HOME"] = "/Users/jm051781/Documents/spark-2.3.1-bin-hadoop2.7/"

# Load training data
data = spark.read.format("libsvm").load("/Users/jm051781/Downloads/class-test2")

# Split the data into train and test
# splits = data.randomSplit([1, 0.4], 1234)
train = data
test = data
train.show()
test.show()
# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial").setFeaturesCol("features")

# train the model
model = nb.fit(train)
# select example rows to display.
predictions = model.transform(test)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))