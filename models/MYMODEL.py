import warnings
import numpy as np
from joblibspark import register_spark
from typing import List

from sklearn.metrics import precision_score, recall_score, confusion_matrix
import torch
from torch import nn
from pyspark.sql.dataframe import DataFrame

warnings.filterwarnings('ignore')
register_spark()

class LeNet5(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=6,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 480, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
            nn.Softmax(dim=1),
        )
    def forward(self, X):
        output1 = self.conv_block1(X)
        output2 = self.conv_block2(output1)
        
        return self.classifier(output2)
    
    def accuracy_fn(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct/len(y_true)) * 100

        return acc
    
class model_LeNet5:
    def __init__(self):
        self.model = LeNet5(input_shape=32, output_shape=10)

    def train(self, df: DataFrame) -> List:
        X = torch.tensor(df.select("image").collect(), dtype=torch.float).reshape(-1,32,32,3)
        y = torch.tensor(df.select("label").collect(), dtype=torch.long).reshape(-1)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=0.1)

        for epoch in range(3):
            self.model.train()

            y_logits = self.model(X)

            y_pred = y_logits.argmax(dim=1)

            loss = loss_fn(y_logits, y)
            acc = self.model.accuracy_fn(y_true = y, y_pred = y_pred)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if epoch == 2:
                print(f"Epoch: {epoch}")
                accuracy = acc
                precision = precision_score(y.cpu().numpy(),y_pred.cpu().numpy(), labels=np.arange(0,10),average="macro")
                recall = recall_score(y.cpu().numpy(),y_pred.cpu().numpy(), labels=np.arange(0,10),average="macro")
                try:
                    f1 = 2*precision*recall/(precision+recall)
                except ZeroDivisionError:
                    f1 = 0.01

                return y_pred, accuracy, precision, recall, f1