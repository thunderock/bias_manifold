
import os
import numpy as np
import pandas as pd


class GloveWrapper(object):
    def __init__(self):
        self.vocab, self.ivocab, self.W, self.b_w, self.U, self.b_u, self.d, self.vocab_path, self.embedding_path = [None] * 9

class Glove():
    def __init__(self, load=False, window_size=8, min_count=5, dim=100, path=None):
        self.window_size = window_size
        self.min_count = min_count
        self.dim = dim
        self._model = GloveWrapper()
        self.eta = 0.05
        if load:
            self.load(path)

    def load_vocab(self, path):
        str2idx, idx2str = dict(), []
        with open(path) as f:
            for (i, line) in enumerate(f):
                entry = line.split()
                str2idx[entry[0]] = (i, int(entry[1]))
                idx2str.append(entry[0])
        return str2idx, idx2str

    def load_bin_vectors(self, embedding_path, vocab_size):
        n = os.path.getsize(embedding_path) // 8
        dim = (n - 2 * vocab_size) // (2 * vocab_size)
        W = np.zeros((vocab_size, dim))
        U = np.zeros((vocab_size, dim))
        b_w = np.zeros(vocab_size)
        b_u = np.zeros(vocab_size)
        with open(embedding_path, 'rb') as f:
            for i in range(vocab_size):
                for j in range(dim):
                    W[i, j] = np.float64(np.frombuffer(f.read(8), dtype=np.float64))
                b_w[i] = np.float64(np.frombuffer(f.read(8), dtype=np.float64))
            for i in range(vocab_size):
                for j in range(dim):
                    U[i, j] = np.float64(np.frombuffer(f.read(8), dtype=np.float64))
                b_u[i] = np.float64(np.frombuffer(f.read(8), dtype=np.float64))
        return W, b_w, U, b_u

    def transform(self, words):
        embs = np.zeros((len(words), self.dim), dtype=np.float32)
        for i, word in enumerate(words):
            if word in self._model.vocab:
                embs[i] = self._model.W[self._model.vocab[word][0]]

        return embs

    def fit(self, lines, workers=4, temp_root='/tmp/'):
        # first write these lines to a file
        df = pd.DataFrame(data=lines, columns=['text'])
        df.to_csv(temp_root + 'temp.txt', index=False, header=False)
        os.system(f'make -C scripts/')
        # creating vocab file
        os.system(f'./scripts/build/vocab_count -min-count {self.min_count} -verbose 2  < {temp_root}temp.txt > {temp_root}temp.vocab')
        overflow_file = temp_root + 'temp_overflow'
        # creating co-occurence file
        os.system(f'./scripts/build/cooccur -memory 6 -vocab-file {temp_root}temp.vocab -verbose 2 -window-size {self.window_size} -overflow-file {overflow_file}< {temp_root}temp.txt > {temp_root}temp.cooccur')
        # creating shuffle file
        os.system(f'./scripts/build/shuffle -memory 6 -verbose 2 -temp-file {temp_root}temp_shuffle.shuf -seed 1 < {temp_root}temp.cooccur > {temp_root}final.shuf')
        # creating glove file
        os.system(f'./scripts/build/glove -save-file {temp_root}glove_trained -threads {workers} -input-file {temp_root}final.shuf -eta {self.eta} -iter 5 -checkpoint-every 0 -vector-size {self.dim} -binary 1 -vocab-file {temp_root}temp.vocab -verbose 2 -seed 1')

        vocab_file = temp_root + 'temp.vocab'
        embedding_file = temp_root + 'glove_trained.bin'
        # self._model.vocab, self._model.ivocab = self.load_vocab(vocab_file)
        # V = len(self._model.vocab)
        # self._model.d = self.window_size
        # self._model.W, self._model.b_w, self._model.U, self._model.b_u = self.load_bin_vectors(embedding_file, V)
        self._model.vocab, _ = self.load_vocab(vocab_file)
        V = len(self._model.vocab)
        _ = self.window_size
        self._model.W, _, _, _ = self.load_bin_vectors(embedding_file, V)
        return self




