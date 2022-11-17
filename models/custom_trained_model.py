
import numpy as np


class CustomTrainedModel():
    """
    only load model
    """
    def __init__(self, path, load_method, load_params, dim=300, ):
        self.load_method = load_method
        self.load_params = load_params
        self.dim = dim
        self.load(path)

    def fit(self, iid, dataset, workers=4):
        print("DummyModel: fit")
        pass

    def save(self, path):
        print("DummyModel: save")
        pass

    def load(self, path):
        self._model = self.load_method(path, **self.load_params)

    def transform(self, words):
        indices = []
        not_found = 0
        for w in words:
            if w in self._model:
                indices.append(self._model[w])
            else:
                not_found += 1
                indices.append(np.zeros(self.dim))
        if not_found > 0:
            print("Not found: ", not_found)
        return np.array(indices)

