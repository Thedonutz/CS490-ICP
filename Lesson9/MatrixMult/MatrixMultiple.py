from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import numpy as np
import os
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, BlockMatrix

os.environ["SPARK_HOME"] = "/Users/jm051781/Documents/spark-2.3.1-bin-hadoop2.7/"

sc = SparkContext()
rows = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]) \
    .zipWithIndex()
rows2 = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) \
    .zipWithIndex()

# need a SQLContext() to generate an IndexedRowMatrix from RDD
sqlContext = SQLContext(sc)
rows = IndexedRowMatrix( \
    rows \
    .map(lambda row: IndexedRow(row[1], row[0])) \
    ).toBlockMatrix()

rows2 = IndexedRowMatrix( \
    rows2 \
    .map(lambda row2: IndexedRow(row2[1], row2[0])) \
    ).toBlockMatrix()

mat_product = rows.multiply(rows2).toLocalMatrix()
print(mat_product)
