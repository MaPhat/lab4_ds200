import numpy as np
from typing import List

class Transforms:
    def __init__(self, transforms : List) -> None:
        self.transforms = transforms
    
    def transform(self, input : np.ndarray) -> np.ndarray:
        for method in self.transforms:
            input = method.transform(input)

        return input