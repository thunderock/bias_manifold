
import gensim
import pickle as pkl


class Word2Vec():

    def __init__(self, load=False, window_size=8, min_count=5, dim=100, path=None):
        self.window_size = window_size
        self.min_count = min_count
        self.dim = dim
        self._model = None
        self.in_vecs = None
        if load:
            self.load(path)

    def fit(self, lines, workers=4):
        model = gensim.models.Word2Vec(window=self.window_size, min_count=self.min_count,
                                             workers=workers, vector_size=self.dim)

        model.build_vocab(lines)
        model.train(lines, total_examples=len(lines), epochs=10)
        self._model = model
        return self._model

    def save(self, path):
        assert self._model is not None, 'Model not fitted yet'
        self._model.save(path)

    def load(self, path):
        try:
            self._model = gensim.models.Word2Vec.load(path)
        except (pkl.UnpicklingError, AttributeError) as e:
            print("There was an error loading the model. Trying to load kv file instead!")
            self.in_vecs = True
            self._model = gensim.models.KeyedVectors.load(path)

    def transform(self, words, WV=None):
        assert isinstance(words, list), 'words should be a list'
        if WV is None:
            if self.in_vecs:
                WV = self._model
            else:
                WV = self._model.wv
        words = [w for w in words if w in WV]
        return WV[words]



