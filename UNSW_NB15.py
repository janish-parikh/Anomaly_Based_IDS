#spark-submit --executor-memory 16g --driver-memory 100g UNSW_NB15.py
import findspark
findspark.init()
import pyspark
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors 
from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import classification_report, confusion_matrix

spark = SparkSession\
     .builder\
     .master("local[10]")\
     .config("spark.executor.memory", "100g")\
     .config("spark.driver.memory", "100g")\
     .config("spark.memory.offHeap.enabled",True)\
     .config("spark.memory.offHeap.size","16g")\
     .config("spark.storage.memoryFraction", "0.4")\
     .appName("sampleCodeForReference")\
     .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.parquet.mergeSchema", "false")
spark.conf.set("spark.hadoop.parquet.enable.summary-metadata", "false")
sc = spark.sparkContext
spark.sparkContext.setLogLevel("ERROR")

#Getting data from csv file
datapath = '/home/ubuntu/janish/UNSW-NB15_1.csv'

data = spark.read.format('csv')\
    .options(header='true',inferschema='true')\
    .load(datapath)
data.show(5, vertical = True)

#Defining Stages of Pipeline
stage_1 = StringIndexer(inputCol= 'state', outputCol= 'state_index')
#for now we are skipping the NAN and null values
stage_1.setHandleInvalid("skip")
# define stage 2: one hot encode the numeric versions of feature 2 and 3 generated from stage 1 and stage 2
stage_2 = OneHotEncoder(inputCols=[stage_1.getOutputCol()], outputCols= ['state_encoded'])

stage_3 = VectorAssembler(inputCols=('spkts', 'dpkts', 'sbytes', 'dbytes','state_encoded',
'rate', 'sttl', 'dttl', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 
'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'), outputCol="features")
# #for now we are skipping the NAN and null values
stage_3.setHandleInvalid("skip")

# # Train a RandomForest model.                     
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=200, maxDepth = 20)

# # Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages= [stage_1, stage_2,stage_3, rf])

# Split the data into training and test sets (20% held out for testing)
(trainingData, testData) = data.randomSplit([0.9, 0.1])

# Train model. This also runs the indexers.
pipelineModel = pipeline.fit(trainingData)

# Make predictions.
predictions = pipelineModel.transform(testData)

# Select example rows to display.
predictions.select("id","prediction","label","attack_cat").show(5, vertical = True)

predictions.write.mode("overwrite").parquet("tmp/output")   

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
print("accuracy is",accuracy)
evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')
print(evaluator.evaluate(predictions))

y_true = predictions.select(['label']).collect()
y_pred = predictions.select(['prediction']).collect()
target_names = ['Normal','Attack']

report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
df = pd.DataFrame(report).transpose()
print(df.head())
df.to_csv('NB15_metrics.csv', index= True)
print(confusion_matrix(y_true, y_pred))
#persisting the pipelinemodel
rfModel = pipelineModel.stages[-1]
print(rfModel)  # summary only
rfr_path = 'model/nb15_test'
pipelineModel.write().overwrite().save(rfr_path)
spark.stop()