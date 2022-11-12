
from torch_geometric.data import download_url
from gensim.corpora.wikicorpus import WikiCorpus
from tqdm import trange

class WikiDataset(object):

    def __init__(self, URL, root='/tmp/'):
        self.file = download_url(URL, root)
        self._lines = False

    @property
    def lines(self):
        if self._lines:
            return self._lines
        corpus = WikiCorpus(self.file, dictionary=True)
        corpus.metadata = True
        articles = list(corpus.get_texts())
        N = len(articles)
        ret = [None] * N
        for i in trange(N):
            article = articles[i][0]
            ret[i] = " ".join(article).lower()
        self._lines = ret
        return self._lines



