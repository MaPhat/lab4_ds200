import pyspark

from pyspark.context import SparkContext
import pyspark.rdd
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT

from transforms import Transforms
from dataloader import Dataloader
from sparkconfig import SparkConfig

class Trainer:
    def __init__(self, model, spark_config : SparkConfig, transforms : Transforms) -> None:
        self.model = model
        self.sparkConf = spark_config
        self.transforms = transforms
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]", f"{self.sparkConf.appname}")
        self.ssc = StreamingContext(self.ssc, self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = Dataloader(self.sc, self.ssc, self.sqlContext, self.sqlContext, self.transforms)
    
    def train(self):
        stream = self.dataloader.parse_stream()
        stream = stream.foreachRDD(self.__train__)

        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, rdd : pyspark.RDD):
        if not rdd.isEmpty():
            schema = StructType(
                [
                    StructField("image", VectorUDT(), True),
                    StructField('label', IntegerType(), True)
                ]
            )
            df = self.sqlContext.createDataFrame(rdd, schema)

            predictions, accuracy, precision, recall, f1 = self.model.train(df)

            print("="*10)
            print(f"Predictions = {predictions}")
            print(f"Accuracy = {accuracy}")
            print(f"Precision = {precision}")
            print(f"Recall = {recall}")
            print(f"F1 Score = {f1}")
            print("="*10)
        
        print("Total Batch Size of RDD Received :",rdd.count())
        print("+"*20)