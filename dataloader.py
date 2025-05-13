"""
Xử lý dữ liệu đầu vào và trả về DStream
"""

import pyspark

from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.dstream import DStream
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.linalg import DenseVector

from transforms import Transforms
from sparkconfig import SparkConfig

import json
import numpy as np
class Dataloader:
    def __init__(self, sc : SparkContext, ssc : StreamingContext, sqlConfig : SQLContext, transforms : Transforms) -> None:
        self.sc = sc
        self.ssc = ssc
        self.sqlConfig = sqlConfig
        self.transforms = transforms
        self.stream = self.ssc.socketTextStream(
            hostname=SparkConfig.stream_host,
            port = SparkConfig.port
        )
    @staticmethod
    def preprossing(stream : DStream, transforms: Transforms) -> DStream:
        stream = stream.map(lambda x : [transforms.transform(x[0]).reshape(-1).tolist(), x[1]])
        stream = stream.map(lambda x: [DenseVector(x[0]), x[1]])
        return stream
    
    def parse_stream(self) -> DStream:
        json_stream = self.stream.map(lambda line : json.loads(line))
        json_stream_exploded = json_stream.flatMap(lambda x : x.values())
        pixels = json_stream_exploded.map(lambda x : [np.array([x[:-1]]).reshape(32,32,3).astype(np.uint8), x[-1]])
        pixels = Dataloader.preprossing(pixels, self.transforms)

        return pixels