import pandas as pd
from ast import literal_eval

class Nyt():

    def __init__(self, path):
        self.path = path
        self._lines = False

    @property
    def lines(self):
        if self._lines:
            return self._lines
        df = pd.read_csv(self.path)
        df.sentence = df.sentence.apply(literal_eval)
        df.sentence = df.sentence.apply(lambda x: " ".join(x).lower())
        self._lines = df.sentence.values.tolist()
        return self._lines