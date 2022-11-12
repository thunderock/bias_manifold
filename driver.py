

from models.word2vec import Word2Vec
from tqdm import tqdm
from utils.weat import WEAT
from utils.dataset import Dataset
from utils.jackknife import JackKnife
import numpy as np
from models.fast_glove import FastGlove
# ! wget -P /tmp/ http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# ! wget -P /tmp/ https://dumps.wikimedia.org/swwiki/latest/swwiki-latest-pages-articles.xml.bz2

# load file
# file = open("../enwik9.txt", "r")
# lines = file.readlines()
# sents = [word_tokenize(line.lower()) for line in tqdm(lines)]

# train the model
# file = '/tmp/swwiki-latest-pages-articles.xml.bz2'
# dataset = object
# ds = Dataset(file)
# model = Word2Vec(load=False)
# model.fit(ds, workers=6)
# model.save("../word2vec.model")


# model = Word2Vec(load=True, path='../word2vec.model')
#
#
# weat = WEAT(model, 'weat/weat.json')
#
# weat_scores = weat.get_scores()
# print(weat_scores)
model = Word2Vec
model = FastGlove
ds = Dataset('../simplewiki-20171103-pages-articles-multistream.xml.bz2')
# print(ds.lines)
jk = JackKnife(ds)
scores = jk.weat_scores(model)

np.save('scores.npy', scores)




