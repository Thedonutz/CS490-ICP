from pyspark.ml.classification import NaiveBayes
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils

import shutil
# Load training data
# from pyspark.python.pyspark.shell import spark
from pyspark import SparkContext

import os

os.environ["SPARK_HOME"] = "/Users/jm051781/Documents/spark-2.3.1-bin-hadoop2.7/"

from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils


sc = SparkContext.getOrCreate()
# Load and parse the data file.
data = MLUtils.loadLibSVMFile(sc, "/Users/jm051781/Downloads/output2")

# Split data approximately into training (60%) and test (40%)
training, test = data.randomSplit([0.8, 0.2])

# Train a naive Bayes model.
model = NaiveBayes.train(training, 1.0)

# Make prediction and test accuracy.
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print('model accuracy {}'.format(accuracy))

# Save and load model
output_dir = 'target/tmp/myNaiveBayesModel'
shutil.rmtree(output_dir, ignore_errors=True)
model.save(sc, output_dir)
sameModel = NaiveBayesModel.load(sc, output_dir)
predictionAndLabel = test.map(lambda p: (sameModel.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print('sameModel accuracy {}'.format(accuracy))