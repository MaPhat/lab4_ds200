from sklearn.svm import LinearSVC
from sklearn.utils import parallel_backend
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from pyspark.sql.dataframe import DataFrame

from typing import List
import numpy as np
class SVM:
    def __init__(self, loss, penalty):
        self.model = LinearSVC(loss=loss, penalty=penalty)

    def train(self, df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select('label').collect()).reshape(-1)

        with parallel_backend("spark", n_jobs=4):
            self.model.fit(X,y)

        predictions = self.model.predict(X)
        predictions = np.array(predictions)

        accuracy = self.model.score(X)
        precision = precision_score(y, predictions, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predictions, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)

        return predictions, accuracy, precision, recall, f1