
class Dataset(object):
    def __init__(self, path, stream=False):
        self.path = path
        self.stream = stream
        self.__lines = None

    def _load_in_array(self):
        from gensim.corpora.wikicorpus import WikiCorpus
        return list(WikiCorpus(self.path).get_texts())

    @property
    def lines(self):
        if self.__lines:
            return self.__lines
        # TODO (ashutiwa): need to handle streams later
        self.__lines = self._load_in_array() if (self.path and not self.stream) else None
        return self.__lines

    @property
    def size(self):
        if not self.__lines:
            self.lines
        return len(self.__lines)


class PandasDataset(Dataset):
    def __init__(self, path, column, stream=False, pickle=False):
        super().__init__(path, stream)
        self.column = column
        self.pickle = pickle

    def _load_in_array(self):
        from ast import literal_eval
        import pandas as pd
        if self.pickle:
            # these are read as lists rather than strings as in case of csv
            return pd.read_pickle(self.path, compression='gzip')[self.column].tolist()

        return pd.read_csv(self.path, usecols=[self.column])[self.column].apply(literal_eval).tolist()

