from trainer import SparkConfig, Trainer
from models import SVM
from transforms import RandomHorizontalFlip, Normalize, Transforms

transforms = Transforms([
    RandomHorizontalFlip(p=0.345), 
    Normalize(
        mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618), 
        std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)
    )
])

if __name__ == "__main__":
    sparkconfig = SparkConfig()

    model = SVM(loss='squared_hinge', penalty='l2')

    trainer = Trainer(model=model, spark_config=sparkconfig, transforms=transforms)

    trainer.train()