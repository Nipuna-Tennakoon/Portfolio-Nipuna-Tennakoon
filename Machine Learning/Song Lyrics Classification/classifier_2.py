#import necessary libraries
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Word2Vec,StringIndexer, Tokenizer, StopWordsRemover
from pyspark.sql.functions import regexp_replace, col
from pyspark.sql.functions import split
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#import medley dataset
df1=pd.read_csv('training/Mendeley dataset.csv',header=0)
df1=df1[['artist_name','track_name','release_date','genre','lyrics']]

#import student_dataset
df2=pd.read_csv('training/Student_dataset.csv',header=0)
df2=df2[['artist_name','track_name','release_date','genre','lyrics']]

#merge datasets
merged_df = pd.concat([df1, df2], ignore_index=True)
merged_df.to_csv("training/Merged_dataset.csv", index=False)

#start the spark session
spark = SparkSession.builder \
    .appName("classifier_2") \
    .getOrCreate()

#import dataset to a dataframe in spark and select necessary columns
dataset=spark.read.csv('training/Merged_dataset.csv',header=True,inferSchema=True)
dataset = dataset.select('lyrics','genre')

#replace single characters and numbers using redular expressions
dataset = dataset.withColumn('lyrics', regexp_replace(col('lyrics'), '[^a-zA-Z\\s]', ''))
dataset = dataset.withColumn("lyrics", regexp_replace(col("lyrics"), r"\d+", ''))

#set the spark transformers and estimator in the pipeline
indexer = StringIndexer(inputCol="genre", outputCol="label")
tokenizer = Tokenizer(inputCol="lyrics", outputCol="tokens")
stop_words_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
word2Vec = Word2Vec(vectorSize=300, minCount=1, inputCol="filtered_tokens", outputCol="lyrics_embedding")
lr = LogisticRegression(featuresCol="lyrics_embedding", labelCol="label")
pipeline = Pipeline(stages=[indexer,tokenizer,stop_words_remover, word2Vec, lr])
train_data, test_data = dataset.randomSplit([0.8, 0.2], seed=42)

#training the model
model = pipeline.fit(train_data)

# make predictions on test_data
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")

# evaluate accuracy on the test set
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy}")

#save the model

model.save("models/classifier_2")

